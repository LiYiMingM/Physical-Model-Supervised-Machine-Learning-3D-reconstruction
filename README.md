# Reliable 3D Reconstruction with Single-Shot Digital Grating and Physical Model-Supervised Machine Learning

## 3D reconstruction flow chart based on MPS_XNet 
<div>
 <img src="./code_wholenet_losspara_optima/images1/main_figure.png" height="240" width="280"/ >
</div>
 The network structure of MPS_XNet 
 <div>
 <img src="./code_wholenet_losspara_optima/images1/psmp_x.png" height="310" width="280"/>
</div>

## 3D reconstruction effect
Prediction on metallic workpieces
<div>
<img src="./code_wholenet_losspara_optima/images1/3D_metal1.png" height="140" width="260"/>
</div>

   
## 🛠️ Environment Setup

The following dependencies are required:

```bash
pip install -r requirements.txt
```
## Training

```bash
python ./code_wholenet_losspara_optima/four_branch_experiment/main_train.py \
```

## Test

```bash
python ./code_wholenet_losspara_optima/four_branch_experiment/main_test.py \
```
