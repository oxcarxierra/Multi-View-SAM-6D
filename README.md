# <p align="center"> <font color=#008000>Multi-view SAM-6D</font>: Improving Zero-Shot 6D Object Pose Estimation via Multi-View Aggregation</p>

Coursework from 3D Vision, Spring 2025, ETH Zürich

Forked from : https://github.com/JiehongLin/SAM-6D

## Version check
Validated with following environment: 
- CUDA 11.8
- python 3.9.6
- pytorch 2.0.0+cu117


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/oxcarxierra/SAM-6D.git
conda create env sam6d python=3.9.6
cd SAM-6D && pip install requirements.txt
```
Install the environment and download the model checkpoints:
```
sh prepare.sh
```

### 2. Evaluation on the custom data
Fix the first lines in demo.sh 
```
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

Seungseok Oh: [littlestein@snu.ac.kr](mailto:littlestein@snu.ac.kr)
