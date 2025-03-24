# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

Forked from : https://github.com/JiehongLin/SAM-6D

## Version check
Validated with following environment: 
- CUDA 12.4
- python 3.9.21
- pytorch 2.6


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/oxcarxierra/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
sh prepare.sh
```

### 2. Evaluation on the custom data
Fix the first lines in demo.sh 
```demo.sh
export CAD_PATH=Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh
```

## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

