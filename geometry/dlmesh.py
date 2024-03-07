import torch

from render import mesh
from render import render
from render import regularizer
from render import util
from torch.cuda.amp import custom_bwd, custom_fwd 
import numpy as np

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad=True)
        self.register_parameter('vertex_pos', self.mesh.v_pos)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        if self.FLAGS.laplace == "absolute":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)
        elif self.FLAGS.laplace == "relative":
            reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)                

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss
    
    def tick_sds(self, glctx, target, lgt, opt_material, iteration, guidance):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers= self.render(glctx, target, lgt, opt_material)
        if self.FLAGS.add_directional_text:
            text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]])
        else:
            text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])
            
        if iteration <= self.FLAGS.coarse_iter:
            srgb =  buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            t = torch.randint( guidance.min_step_early, guidance.max_step_early+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
        else:
            srgb =   buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            t = torch.randint( guidance.min_step_late, guidance.max_step_late+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]

        pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        latents = guidance.encode_imgs(pred_rgb_512)
       
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states= text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond)

        if iteration <= self.FLAGS.coarse_iter:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        else:
            w = 1 / (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad = w* (noise_pred -noise) 
        grad = torch.nan_to_num(grad)
        sds_loss = SpecifyGradient.apply(latents, grad)
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        # reg_loss = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) *100
     
        return sds_loss, reg_loss
    
    def tick_rdl(self, glctx, target, lgt, opt_material, iteration, guidance):

        buffers= self.render(glctx, target, lgt, opt_material)
        if self.FLAGS.add_directional_text:
            text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]])
            ori_text_z = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.ori_text_z[target['prompt_index']]])
        else:
            text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])
            ori_text_z = torch.cat([guidance.uncond_z, guidance.ori_text_z])
            
        if iteration <= self.FLAGS.coarse_iter:
            srgb =  buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            t = torch.randint(guidance.min_step_early, guidance.max_step_early+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
        else:
            srgb =   buffers['shaded'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            t = torch.randint(guidance.min_step_late, guidance.max_step_late+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]

        pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        latents = guidance.encode_imgs(pred_rgb_512)
       
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states= text_embeddings).sample
            ori_noise_pred = guidance.unet(latent_model_input, tt, encoder_hidden_states= ori_text_z).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond)

        noise_pred_uncond, noise_pred_text = ori_noise_pred.chunk(2)
        ori_noise_pred = noise_pred_text + guidance.guidance_weight * (noise_pred_text - noise_pred_uncond)

        if iteration <= self.FLAGS.coarse_iter:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        else:
            w = 1 / (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad = w* (noise_pred -ori_noise_pred) 
        # grad = w* (0.6*noise_pred + 0.4*ori_noise_pred-noise) 
        grad = torch.nan_to_num(grad)
        sds_loss = SpecifyGradient.apply(latents, grad)
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        # reg_loss = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) *100
     
        return sds_loss, reg_loss
