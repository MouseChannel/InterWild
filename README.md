## Introduction
This is the PyTorch portion of the **[Project](https://github.com/MouseChannel/MCHand)**, where some content has been modified, and it has successfully been converted into a TensorRT engine.

origin Paper is **[Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild (CVPR 2023)](https://arxiv.org/abs/2303.13652)**. 


origin implementation is **[https://github.com/facebookresearch/InterWild](https://github.com/facebookresearch/InterWild)**



## Setup
- install environment
- download [CheckPoint](https://drive.google.com/file/d/12temUVaIhrpUqw-zzXArqI6cm5aMfVWa/view) snapshot_6.pth
- `python mousechannelExport.py` to Get mousechannelExport.onnx