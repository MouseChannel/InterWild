import os.path as osp

import torch
from main.model import get_model
from torch.nn.parallel.data_parallel import DataParallel
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from main.config import cfg
from main.model import get_model
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image

from common.utils.vis import vis_keypoints_with_skeleton, save_obj
from common.utils.mano import mano
model_path = './demo/snapshot_6.pth'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

original_img = load_img('./demo/images/image.jpg')
img_height, img_width = original_img.shape[:2]
bbox = [0, 0, img_width, img_height]
bbox = process_bbox(bbox, img_width, img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
img = torch.from_numpy(img).float().permute(2, 0, 1) / 255
img = img[None, :, :, :].cuda()

torch.onnx.export(model.module,  # model being run
                  img,  # model input (or a tuple for multiple inputs)
                  f="mousechannelExport.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file

                  opset_version=17,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names

                  output_names=[
                      'rjoint_img',
                      'ljoint_img',
                      'rmano_mesh',
                      'lmano_mesh'
                  ],

                  )
