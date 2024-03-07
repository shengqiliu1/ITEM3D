# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import argparse
import json
import torch
import nvdiffrast.torch as dr

# Import data readers / generators
from dataset.dataset_item3d import DatasetItem3D

# Import topology / geometry trainers
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import util
from render import mesh
from render import light
from render_utils import initial_guess_material

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    shape = [target['mv'].shape[0],] + target['resolution'] + [3,]
    if bg_type == 'black':
        background = torch.zeros(shape, dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(shape, dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['background'] = background

    return target

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--train_net', type=bool, default=False)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-dt', '--directional-text', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='white', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument("--front_threshold", type= int, nargs=1, default= 45 , help="the range of front view would be [-front_threshold, front_threshold")
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 128                      # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "absolute"               # Mesh Laplacian ["absolute", "relative", "large_steps"]
    FLAGS.laplace_scale       = 10000                  # Weight for sdf regularizer. Default is relative with large weight
    # FLAGS.normal_scale        = 0.02                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if FLAGS.multi_gpu:

        # adjust total iters
        FLAGS.iter = int(FLAGS.iter / int(os.environ["WORLD_SIZE"]))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            if key != 'out_dir':
                FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    # if FLAGS.out_dir is None:
    #     FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    # else:
    #     FLAGS.out_dir = 'test/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    # os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeCudaContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_mesh = mesh.load_mesh(FLAGS.base_mesh)
    geometry = DLMesh(base_mesh, FLAGS)
    lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
    mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)

    dataset_validate = DatasetItem3D(glctx, FLAGS, validate=True, gif=False)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    for itr, target in enumerate(dataloader_validate):
        target = prepare_batch(target, FLAGS.background)
        buffers = geometry.render(glctx, target, lgt, mat)

        # if use_normal:
        #     result_dict['opt'] = util.rgb_to_srgb(buffers['normal'][...,0:3])[0]
        # else:
        #     result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        # result_image = result_dict['opt']

        rgb = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        # normal = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
        os.makedirs("%s/source" % (FLAGS.save_dir), exist_ok=True)
        util.save_image("%s/source/val_%06d.png" % (FLAGS.save_dir,itr), rgb.cpu().detach().numpy())
        # util.save_image("%s/source/val_normal_%06d.png" % (FLAGS.save_dir,itr), normal.cpu().detach().numpy())
