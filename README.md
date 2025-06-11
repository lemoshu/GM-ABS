# GM-ABS


This repository holds the PyTorch implementation of the paper [GM-ABS: Promptable Generalist Model Drives Active Barely Supervised Training in Specialist Model for 3D Medical Image Segmentation](). 

## Introduction
### Abstract
Semi-supervised learning (SSL) has greatly advanced 3D medical image segmentation by alleviating the need for intensive labeling by radiologists. While previous efforts focused on \textit{model-centric} advancements, the emergence of foundational generalist models like the Segment Anything Model (SAM) is expected to reshape the SSL landscape. Although these generalists usually show performance gaps relative to previous specialists in medical imaging, they possess impressive zero-shot segmentation abilities with manual prompts. Thus, this capability could serve as ``free lunch" for training specialists, offering future SSL a promising \textit{data-centric} perspective, especially revolutionizing both pseudo and expert labeling strategies to enhance the data pool. In this regard, we propose the Generalist Model-driven Active Barely Supervised (GM-ABS) learning paradigm, for developing specialized 3D segmentation models under extremely limited (barely) annotation budgets, e.g., merely cross-labeling three slices per selected scan. In specific, building upon a basic mean-teacher SSL framework, GM-ABS modernizes the SSL paradigm with two key data-centric designs: (i) Specialist-generalist collaboration, where the in-training specialist leverages class-specific positional prompts derived from class prototypes to interact with the frozen class-agnostic generalist across multiple views to achieve noisy-yet-effective label augmentation. Then, the specialist robustly assimilates the augmented knowledge via noise-tolerant collaborative learning. (ii) Expert-model collaboration that promotes active cross-labeling with notably low labeling efforts. This design progressively furnishes the specialist with informative and efficient supervision via a human-in-the-loop manner, which in turn benefits the quality of class-specific prompts. Extensive experiments on three benchmark datasets highlight the promising performance of GM-ABS over recent SSL approaches under extremely constrained labeling resources.

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

If you find this paper useful, please cite it as:
```
@article{xu2024FMABS,
  title={FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation},
  author={Xu, Zhe and Chen, Cheng and Lu, Donghuan and Sun, Jinghan and Wei, Dong and Zheng, Yefeng and Li, Quanzheng and Tong, Raymond Kai-yu},
  journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024}
}
```
