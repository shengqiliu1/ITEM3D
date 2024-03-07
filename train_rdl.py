# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import argparse
import json
import math
import numpy as np
import torch
import nvdiffrast.torch as dr

# Import data readers / generators
from dataset.dataset_item3d import DatasetItem3D
from dataset.dataset_item3d import get_camera_params

# Import topology / geometry trainers
from geometry.dlmesh import DLMesh


from render import obj
from render import util
from render import mesh
from render import light
from stable_diffusion.sd_util import StableDiffusion
from tqdm import tqdm
import torchvision.transforms as transforms
from render import util
from render.video import Video
import random
import imageio
import os.path as osp
from render_utils import initial_guess_material, xatlas_uvmap, save_uvmap, get_normalize_mesh

@torch.no_grad()
def prepare_batch(target, background= 'black'):
    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['normal_rotate'] = target['normal_rotate'].cuda()
    # target['prompt_index'] = target['prompt_index'].cuda()
    batch_size = target['mv'].shape[0]
    resolution = target['resolution']
    if background == 'white':
        target['background']= torch.ones(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device='cuda') 
    if background == 'black':
        target['background'] = torch.zeros(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device='cuda')

    return target

@torch.no_grad()  
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, relight = None):
    result_dict = {}
    with torch.no_grad():
        with torch.no_grad():
            lgt.build_mips()
            if FLAGS.camera_space_light:
                lgt.xfm(target['mv'])
            if relight != None:
                relight.build_mips()
        buffers = geometry.render(glctx, target, lgt, opt_material)
        result_dict['shaded'] =  buffers['shaded'][0, ..., 0:3]
        result_dict['shaded'] = util.rgb_to_srgb(result_dict['shaded'])
        if relight != None:
            result_dict['relight'] = geometry.render(glctx, target, relight, opt_material)['shaded'][0, ..., 0:3]
            result_dict['relight'] = util.rgb_to_srgb(result_dict['relight'])
        result_dict['mask'] = (buffers['shaded'][0, ..., 3:4])
        result_image = result_dict['shaded']

        if FLAGS.display is not None :
            # white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'bsdf' in layer:
                    buffers  = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)

        return result_image, result_dict

def save_gif(dir,fps):
    imgpath = dir
    frames = []
    for idx in sorted(os.listdir(imgpath)):
        # print(idx)
        img = osp.join(imgpath,idx)
        frames.append(imageio.v2.imread(img))
    imageio.mimsave(os.path.join(dir, 'eval.gif'),frames,'GIF',duration=1/fps)
    
@torch.no_grad()     
def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS, relight= None):
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    
    shaded_dir = os.path.join(out_dir, "shaded")
    relight_dir = os.path.join(out_dir, "relight")
    kd_dir = os.path.join(out_dir, "kd")
    ks_dir = os.path.join(out_dir, "ks")
    normal_dir = os.path.join(out_dir, "normal")
    mask_dir = os.path.join(out_dir, "mask")
    
    os.makedirs(shaded_dir, exist_ok=True)
    os.makedirs(relight_dir, exist_ok=True)
    os.makedirs(kd_dir, exist_ok=True)
    os.makedirs(ks_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print("Running validation")
    dataloader_validate = tqdm(dataloader_validate)
    for it, target in enumerate(dataloader_validate):

        # Mix validation background
        target = prepare_batch(target, 'white')

        result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, relight)
        for k in result_dict.keys():
            np_img = result_dict[k].detach().cpu().numpy()
            if k == 'shaded':
                util.save_image(shaded_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'relight':
                util.save_image(relight_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'kd':
                util.save_image(kd_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'ks':
                util.save_image(ks_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'normal':
                util.save_image(normal_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'mask':
                util.save_image(mask_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
    return 0

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, FLAGS, guidance):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.FLAGS = FLAGS
        self.guidance = guidance
        self.if_flip_the_normal = FLAGS.if_flip_the_normal
        self.if_use_bump = FLAGS.if_use_bump
        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []
      

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])
        return self.geometry.tick_dds(glctx, target, self.light, self.material, it , self.guidance)

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    log_interval=10,
    optimize_light=True,
    optimize_geometry=True,
    guidance = None
    ):
    
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=False)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
   
    model = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, FLAGS, guidance)
    if optimize_geometry:
        optimizer_mesh = torch.optim.AdamW(model.geo_params, lr=0.001, betas=(0.9, 0.99), eps=1e-15)
    optimizer = torch.optim.AdamW(model.params, lr=0.01, betas=(0.9, 0.99), eps=1e-15)

    if FLAGS.multi_gpu: 
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[FLAGS.local_rank], find_unused_parameters=True)
        
    img_cnt = 0
    sds_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []
   

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)
    scaler = torch.cuda.amp.GradScaler(enabled=True)  
    
    rot_ang = 0
    if FLAGS.local_rank == 0:
        video = Video(FLAGS.out_dir)
    if FLAGS.local_rank == 0:
        dataloader_train = tqdm(dataloader_train)
    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        target = prepare_batch(target, FLAGS.train_background)  

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            save_video = FLAGS.video_interval and (it % FLAGS.video_interval == 0)
            if  save_image:
                result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.train_background), geometry, opt_material, lgt, FLAGS)  #prepare_batch(next(v_it), FLAGS.background)
                np_result_image = result_image.detach().cpu().numpy()
                util.save_image(FLAGS.out_dir + '/' + ('img_%06d.png' % (img_cnt)), np_result_image)
                img_cnt = img_cnt+1
            if save_video:
                with torch.no_grad():
                    params = get_camera_params(
                                resolution=512,
                                fov=45,
                                elev_angle=-20,
                                azim_angle =rot_ang,       
                            )  
                    rot_ang += 1
                    buffers  = geometry.render(glctx, params, lgt, opt_material, bsdf='pbr')
                    video_image = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                    video_image = video.ready_image(video_image)
                    
        iter_start_time = time.time()
        
        with torch.cuda.amp.autocast(enabled= True):
            sds_loss, reg_loss = model(target, it)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = reg_loss + sds_loss
        
        scaler.scale(total_loss).backward()
        sds_loss_vec.append(sds_loss.item())
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        scaler.step(optimizer)
        optimizer.zero_grad()

        if optimize_geometry:
            scaler.step(optimizer_mesh)
            # scheduler_mesh.step()
            optimizer_mesh.zero_grad()
                

        scaler.update()
        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)
    return geometry, opt_material

def seed_everything(seed, local_rank):
    random.seed(seed + local_rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-si', '--save-interval', type=int, default=1000, help="The interval of saving an image")
    parser.add_argument('-vi', '--video_interval', type=int, default=10, help="The interval of saving a frame of the video")
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--train_background', default='black', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--add_directional_text", action='store_true', default=False)
    parser.add_argument('--text', type=str, default="", help="text prompt")
    parser.add_argument('--ori_text', type=str, default="", help="origin text prompt")
    parser.add_argument('--camera_random_jitter', type= float, default=0.4, help="A large value is advantageous for the extension of objects such as ears or sharp corners to grow.")
    parser.add_argument('--fovy_range', nargs=2, type=float, default=[25.71, 45.00])
    parser.add_argument('--elevation_range', nargs=2, type=int, default=[-10, 45], help="The elevatioin range must in [-90, 90].")
    parser.add_argument("--guidance_weight", type=int, default=100, help="The weight of classifier-free guidance")
    parser.add_argument("--translation_y", type= float, nargs=1, default= 0 , help="translation of the initial shape on the y-axis")
    parser.add_argument("--translation_z", type= float, nargs=1, default= 0 , help="translation of the initial shape on the z-axis")
    parser.add_argument("--coarse_iter", type= int, nargs=1, default= 1000 , help="The iteration number of the coarse stage.")
    parser.add_argument('--early_time_step_range', nargs=2, type=float, default=[0.02, 0.5], help="The time step range in early phase")
    parser.add_argument('--late_time_step_range', nargs=2, type=float, default=[0.02, 0.5], help="The time step range in late phase")
    parser.add_argument("--if_flip_the_normal", action='store_true', default=False , help="Flip the x-axis positive half-axis of Normal. We find this process helps to alleviate the Janus problem.")
    parser.add_argument("--front_threshold", type= int, nargs=1, default= 45 , help="the range of front view would be [-front_threshold, front_threshold")
    parser.add_argument("--if_use_bump", type=bool, default= True , help="whether to use perturbed normals during appearing modeling")
    parser.add_argument("--uv_padding_block", type= int, default= 4 , help="The block of uv padding.")
    parser.add_argument("--negative_text", type=str, default="", help="adding negative text can improve the visual quality in appearance modeling")
    FLAGS = parser.parse_args()
    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64, 128 and 256 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.relight             = None                     # HDR environment probe(relight)
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [1, 50]
    FLAGS.learn_light         = False
    FLAGS.gpu_number          = 1
    # FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
     
    if FLAGS.multi_gpu:
        FLAGS.gpu_number = int(os.environ["WORLD_SIZE"])
        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", world_size = FLAGS.gpu_number, rank = FLAGS.local_rank)  
        torch.cuda.set_device(FLAGS.local_rank)
  
    
    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    seed_everything(FLAGS.seed, FLAGS.local_rank)
    
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    # glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext()
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    dataset_train    = DatasetItem3D(glctx, FLAGS, validate=False)
    dataset_validate = DatasetItem3D(glctx, FLAGS, validate=True)
    dataset_gif      = DatasetItem3D(glctx, FLAGS, gif=True)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=1)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    guidance = StableDiffusion(device = 'cuda',
                               text = FLAGS.text,
                               ori_text= FLAGS.ori_text, 
                               add_directional_text = FLAGS.add_directional_text,
                               batch = FLAGS.batch,
                               guidance_weight = FLAGS.guidance_weight,
                               sds_weight_strategy = FLAGS.sds_weight_strategy,
                               early_time_step_range = FLAGS.early_time_step_range,
                               late_time_step_range= FLAGS.late_time_step_range,
                               negative_text = FLAGS.negative_text)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad_(False)

    if FLAGS.base_mesh is None:
        assert False, "[Error] The path of custom mesh is invalid ! (appearance modeling)"
    # Load initial guess mesh from file
    base_mesh = mesh.load_mesh(FLAGS.base_mesh)
    geometry = DLMesh(base_mesh, FLAGS)

    # mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)
    mat = initial_guess_material(geometry, True, FLAGS)
    geometry, mat = optimize_mesh(glctx, 
                                    geometry, 
                                    mat, 
                                    lgt, 
                                    dataset_train, 
                                    dataset_validate, 
                                    FLAGS, 
                                    optimize_light=FLAGS.learn_light,
                                    optimize_geometry= False, 
                                    guidance= guidance,
                                    )
    
    # ==============================================================================================
    #  Validate
    # ==============================================================================================
    if FLAGS.validate and FLAGS.local_rank == 0:
        if FLAGS.relight != None:       
            relight = light.load_env(FLAGS.relight, scale=FLAGS.env_scale)
        else:
            relight = None
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS, relight)
    if FLAGS.local_rank == 0:
        if 'kd_ks_normal' in mat.keys():
            base_mesh = save_uvmap(glctx, geometry, mat, FLAGS)
        else:
            base_mesh = geometry.getMesh(mat)
    torch.cuda.empty_cache()
    if 'kd_ks_normal' in mat.keys():
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']
    lgt = lgt.clone()
    if FLAGS.local_rank == 0:
        os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt) 
