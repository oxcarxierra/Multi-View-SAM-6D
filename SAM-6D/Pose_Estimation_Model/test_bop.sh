#!/bin/bash

#SBATCH --time=300:00
#SBATCH --account=3dv
#SBATCH --output=test_bop.out

python test_bop_multiview.py --n_multiview 5 --exp_id 0 --visualization=True