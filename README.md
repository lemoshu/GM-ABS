# GM-ABS


This repository holds the PyTorch implementation of the paper [GM-ABS: Promptable Generalist Model Drives Active Barely Supervised Training in Specialist Model for 3D Medical Image Segmentation](). 

## Introduction
### Abstract
Semi-supervised learning (SSL) has significantly advanced 3D medical image segmentation by reducing the need for intensive expert annotation. While most prior work has emphasized **model-centric** innovations, the rise of foundation generalist models such as Segment Anything Model (SAM) promises to reshape SSL from a **data-centric** perspective. Despite a performance gap compared to specialist models in the medical domain, generalist models offer impressive zero-shot capabilities through manual prompting—an untapped source of “free lunch” for training specialists.

To this end, we propose **GM-ABS**: a **Generalist Model-driven Active Barely Supervised** learning framework for building high-quality specialist segmentation models under extremely limited annotation budgets (e.g., cross-labeling only three slices per scan). GM-ABS extends a basic mean-teacher SSL framework with two novel, data-centric designs:

1. **Specialist–Generalist Collaboration**: The in-training specialist interacts with a frozen class-agnostic generalist model using class-specific prompts derived from learned prototypes. This enables multi-view, noisy-but-useful label augmentation, which the specialist learns via a noise-tolerant consistency objective.
2. **Expert–Model Collaboration**: A human-in-the-loop active learning strategy facilitates efficient and informative cross-slice labeling, boosting both supervision efficiency and the quality of future prompts.

Extensive experiments on three benchmark datasets show that GM-ABS consistently outperforms recent SSL baselines under extremely limited labeling resources.

## :hammer: Requirements
Set up the conda environment.

Please refer to `requirements.txt`. Key dependencies include:

- `torch==1.10.1+cu113`
- `Python==3.6.5`
- `cleanlab==2.2.0`

Ensure CUDA compatibility with your system.

## ✅ To-do List
- ✅ Basic implementation
- ☐ Baseline methods integration
- ☐ Compatibility with other SSL frameworks
- ☐ Multi-dataset coverage


## :computer: Instruction

1. Preparation
- **Cross-labeling setup**: See [`./dataloaders/cross_labeling.py`](./code_GMABS/dataloaders/cross_labeling.py) for details.
- **Generalist model weights**: Place pre-trained generalist models in `./sam_weights/`. Examples:
  - [MobileSAM weights](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights)
  - [MedSAM / LiteMedSAM](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN)

You should follow their official instruction to install the package. For [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/tree/master), you can run:
Install Mobile Segment Anything:
```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```


2. Train
```
python train_final_GMABS_LA_public.py \
    --labeled_num 4 \
    --budget 16 \
    --active_type uncerper_div \
    --gpu 0 \
    --label_strategy majority \
    --exp LA_GMABS_HERD \
    --add_point 2
```


3. Test 
```
cd ./code
python test_3D.py --model vnet_AL --gpu 0
```


## :books: Citation

If you find this work helpful, please cite us:
```
@inproceedings{xu2024FMABS,
  title={FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation},
  author={Xu, Zhe and Chen, Cheng and Lu, Donghuan and Sun, Jinghan and Wei, Dong and Zheng, Yefeng and Li, Quanzheng and Tong, Raymond Kai-yu},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024}
}
```
