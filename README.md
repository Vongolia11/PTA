# Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher

This repository contains the research code for **Purify-then-Align (PTA)**, a multimodal human sensing framework for robust learning under missing modalities.

PTA combines two main ideas:

- **meta-weighted multimodal learning**, which learns modality importance and reduces the influence of weak/noisy modalities;
- **diffusion-based knowledge distillation**, which transfers cross-modal knowledge to strengthen unimodal representations.

The repository corresponds to the **CVPR Workshop version** of the project.

---

## Important note on XRF55 / HAR reproduction

The **XRF55/HAR branch** in this repository was reorganized from an earlier research codebase after the project was completed. Some original experimental artifacts and checkpoints are no longer available, and several packaging issues were introduced during the later reorganization.

We have corrected the obvious packaging issues in the current release, but **we cannot guarantee exact reproduction of the XRF55 numbers reported in Table 2** from this repository.

For the currently released XRF55 implementation, the following points are useful:

- the pretrained modality encoders are **fine-tuned end-to-end** in the released training pipeline;
- to the best of our recollection, the prepared `.npy` inputs correspond to the official XRF55 data and no additional normalization/scaling/clipping is applied inside the PTA training pipeline;
- Scene 1 is used, with repetitions/trials **1–14 for training** and **15–20 for testing**;
- the training set is further divided approximately **80/20** for the bi-level optimization:
  - 80% for optimizing the main network;
  - 20% for the outer-loop optimization of modality weights;
- the reported XRF55 result was obtained from a **single run**;
- the intended configuration uses 30,000 iterations, an initial learning rate of `2e-4`, batch size 32, and a distillation coefficient of `0.1`.

The bi-level modality-weighting design was inspired by:

- **Meta-Learned Modality-Weighted Knowledge Distillation for Robust Multi-Modal Learning with Missing Data**  
  https://arxiv.org/abs/2405.07155

The diffusion-based feature distillation component was inspired by:

- **Knowledge Diffusion for Distillation**  
  https://arxiv.org/abs/2305.15712

### Recommended newer implementation: COMPASS

If your primary interest is **missing-modality multimodal sensing, especially XRF55/HAR**, we also recommend our more recent project **COMPASS**:

**https://github.com/haowangcoder/COMPASS**

COMPASS is a newer project with a better-preserved public release, including explicit dataset preparation, training commands, environment information, and pretrained checkpoints. It may therefore be a better starting point for new experiments and reproducibility studies on XRF55.

---

## Highlights

PTA supports two tasks:

- **HPE:** Human Pose Estimation on **MM-Fi**
- **HAR:** Human Action Recognition on **XRF55**

Core components include:

- meta-weight learning / modality weighting;
- diffusion-based knowledge distillation / alignment;
- task-specific modality encoders and prediction heads.

This release intentionally remains close to the original supplementary-material code structure. Some script names and internal identifiers may therefore reflect legacy research code.

---

## Repository structure

```text
PTA/
├── DualNet.py
├── Encoders.py
├── engine.py
├── Extractor.py
├── logger.py
├── misc.py
├── Task.py
├── HAR/
│   ├── train.py
│   ├── eval_all.py
│   ├── split_train_test.py
│   ├── HAR_Task.py
│   ├── XRF55_Dataset.py
│   ├── utils.py
│   ├── backbone_models/
│   └── losses/
│       ├── dist_kd.py
│       ├── kd_loss.py
│       ├── kl_div.py
│       └── diffkd/
└── HPE/
    ├── train.py
    ├── evaluate.py
    ├── eval2.py
    ├── task.py
    ├── syn_DI_dataset.py
    ├── utils.py
    ├── config.yaml
    ├── backbones/
    └── meta_diffusion/
        └── losses/
            ├── dist_kd.py
            ├── kd_loss.py
            ├── kl_div.py
            └── diffkd/
```

---

## Environment

This codebase is implemented in Python and PyTorch.

Typical dependencies include:

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- SciPy
- PyYAML
- tqdm
- tensorboardX

A minimal installation example is:

```bash
pip install torch torchvision numpy scipy pyyaml tqdm tensorboardX
```

Because this repository is released close to the original research environment, package-version adjustments may be required depending on your CUDA/PyTorch setup.

---

## Data and pretrained backbones

For dataset download and the original backbone setup, we recommend following the **X-Fi** repository:

https://github.com/xyanchen/X-Fi

### HPE: MM-Fi

Download MM-Fi and prepare the pretrained backbone weights following the X-Fi instructions.

A typical layout is:

```text
Data/
└── MM-Fi/
    ├── P01/
    └── ...

HPE/
└── backbones/
    ├── RGB_benchmark/
    │   └── RGB_Resnet18.pt
    ├── depth_benchmark/
    │   └── depth_Resnet18.pt
    ├── mmwave_benchmark/
    │   └── mmwave_all_random_TD.pt
    ├── lidar_benchmark/
    │   └── lidar_all_random.pt
    └── CSI_benchmark/
        └── protocol3_random_1.pkl
```

### HAR: XRF55

The HAR scripts use paths relative to the `HAR/` working directory. Place the official XRF55 data under:

```text
HAR/
└── Data/
    └── XRF55_Dataset/
        └── Scene1/
            ├── RFID/
            ├── WiFi/
            └── mmWave/
```

Place the pretrained modality encoders under:

```text
HAR/
└── backbone_models/
    ├── mmWave/
    │   └── mmwave_ResNet18.pt
    ├── WIFI/
    │   └── wifi_ResNet18.pt
    └── RFID/
        └── rfid_ResNet18.pt
```

Then create the train/test split:

```bash
cd HAR
python split_train_test.py
```

This creates:

```text
HAR/Data/Split_XRF55_Dataset/
├── train_data/Scene1/
└── test_data/Scene1/
```

The split script assigns repetitions/trials 1–14 to training and 15–20 to testing.

---

## Running the code

### HPE: MM-Fi

Review `HPE/config.yaml` first.

```bash
cd HPE
python train.py --dataset ../Data/MM-Fi
```

Evaluation:

```bash
python eval2.py
```

### HAR: XRF55

After preparing the data and pretrained backbones:

```bash
cd HAR
python train.py
```

The released training code further splits the XRF55 training partition into an inner-loop subset and an outer-loop/meta subset.

Evaluation example:

```bash
python eval_all.py \
  --data_dir ./Data/Split_XRF55_Dataset \
  --reload_path ./checkpoint/example/xrf55_last.pth
```

---

## Method components

### Meta-weighted fusion / weighting logic

- `Task.py`
- `HPE/task.py`
- `HAR/HAR_Task.py`
- `DualNet.py`

### Diffusion-based knowledge distillation

- `HPE/meta_diffusion/losses/`
- `HAR/losses/`

### Task-specific backbones and heads

- `HPE/backbones/`
- `HAR/backbone_models/`
- `Extractor.py`

---

## Reproducibility checklist

Before training, please verify:

- the correct dataset is downloaded;
- the expected pretrained modality backbones are available;
- all paths are correct relative to the script working directory;
- `split_train_test.py` has been run for XRF55;
- your PyTorch/CUDA environment is compatible;
- for XRF55, the released implementation fine-tunes the pretrained encoders.

For new XRF55 experiments, we additionally recommend comparing against or starting from the more recent **COMPASS** release:

https://github.com/haowangcoder/COMPASS

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{weng2026pta,
  title     = {Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher},
  author    = {Weng, Pengcheng and Qian, Yanyu and Xu, Yangxin and Wang, Fei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2026},
  note      = {Accepted to CVPR 2026 Workshops}
}
```

---

## Acknowledgements

We thank the authors of **X-Fi** for the dataset/backbone reference implementation and setup pipeline:

https://github.com/xyanchen/X-Fi

The meta-learning and diffusion-distillation components were influenced by:

- https://arxiv.org/abs/2405.07155
- https://arxiv.org/abs/2305.15712

We also point readers interested in a newer, more reproducible missing-modality sensing implementation to:

- https://github.com/haowangcoder/COMPASS

---

## Contact

For questions about PTA, please open an issue in this repository.
