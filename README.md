# FM-ABS


This repository will hold the PyTorch implementation of the paper [MICCAI24](). 

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
cd ./AC-MT
```

2. Data Preparation
Refer to ./data for details


3. Train
```
cd ./code
python train_ACMT_{}_3D.py --labeled_num {} --gpu 0
```

4. Test 
```
cd ./code
python test_3D.py
```


## :books: Citation

If you find this paper useful, please cite as:
```
@article{xu2023ambiguity,
  title={Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation},
  author={Xu, Zhe and Wang, Yixin and Lu, Donghuan and Luo, Xiangde and Yan, Jiangpeng and Zheng, Yefeng and Tong, Raymond Kai-yu},
  journal={Medical Image Analysis},
  pages={102880},
  year={2023},
  publisher={Elsevier}
}
```
