# FM-ABS


This repository will hold the PyTorch implementation of the MICCAI'24 paper [FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation]() and the journal extension [Journal](). 

[Note] Under construction. The entire project will be released upon the journal extension acceptance. The repository will be moved to [New Repo](https://github.com/lemoshu/GM-ABS)

## Introduction
### Abstract
Placeholder

## :hammer: Requirements
Check requirements.txt.
* Pytorch version >=0.4.1.
* Python == 3.6 

## :computer: Usage

1. Clone the repo:
```
cd ./FM-ABS
```

2. Data Preparation
Refer to ./data for details


3. Train
```
cd ./code
python train_FMABS_{}_3D.py --labeled_num {} --gpu 0
```

4. Test 
```
cd ./code
python test_3D.py
```


## :books: Citation

If you find this paper useful, please cite as:
```
@article{xu2024FMABS,
  title={FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation},
  author={Xu, Zhe and Chen, Cheng and Lu, Donghuan and Sun, Jinghan and Wei, Dong and Zheng, Yefeng and Li, Quanzheng and Tong, Raymond Kai-yu},
  journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024}
}
```
