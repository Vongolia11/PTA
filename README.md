# Purify-then-Align:Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher

This repository contains the research code for our multimodal human sensing project on **robust learning under missing modalities**.

The method combines two key ideas:

- **meta-weighted multimodal fusion**, which learns to down-weight weak or noisy modalities;
- **diffusion-based knowledge distillation**, which transfers cross-modal knowledge to strengthen unimodal encoders.

This release follows the **original supplementary-material code structure** with minimal refactoring. As a result, some folder names, script names, and internal identifiers preserve **legacy naming** from an earlier submission stage. The repository corresponds to the **CVPR Workshop version** of the project.

---

## Highlights

- Supports two tasks:
  - **HPE**: Human Pose Estimation on **MM-Fi**
  - **HAR**: Human Action Recognition on **XRF55**
- Includes the core components of the method:
  - **meta-weight learning / modality weighting**
  - **diffusion-based KD / alignment**
  - task-specific backbones and heads
- Released in a form close to the original research code for easier comparison with the paper and supplementary material

---

## Repository Structure

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
│   ├── readme.md
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
    ├── readme.md
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

This codebase is implemented in **Python** and **PyTorch**.

Typical dependencies include:

- Python 3.8+
- PyTorch
- torchvision
- numpy
- scipy
- pyyaml
- tqdm
- tensorboardX

A minimal installation example is:

```bash
pip install torch torchvision numpy scipy pyyaml tqdm tensorboardX
```

Since this repository is released close to the original research environment, you may need to adjust package versions based on your local CUDA / PyTorch setup.

---

## Data and Pretrained Weights

For dataset download and the original backbone setup, we recommend directly following the **X-Fi** repository:

- **X-Fi GitHub**: https://github.com/xyanchen/X-Fi

This repository assumes a directory organization compatible with that setup.

### HPE branch: MM-Fi

Please download **MM-Fi** and prepare the pretrained backbone weights following the instructions from the X-Fi repository.

Expected layout:

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

### HAR branch: XRF55

Please download **XRF55** and prepare the pretrained encoders. The directory structure should look like:

```text
Data/
└── XRF55_Dataset/
    ├── Scene1/
    └── ...

HAR/
└── backbone_models/
    ├── mmWave/
    │   └── mmwave_ResNet18.pt
    ├── WIFI/
    │   └── wifi_ResNet18.pt
    └── RFID/
        └── rfid_ResNet18.pt
```

Then preprocess the raw XRF55 data:

```bash
cd HAR
python split_train_test.py
cd ..
```

This will create a processed split under `Data/Split_XRF55_Dataset/`.

---

## Running the Code

### HPE: MM-Fi Human Pose Estimation

Review `HPE/config.yaml` first. Important options include:

- modality combination
- protocol
- data split
- batch size
- learning rate
- meta learning rate
- KD loss weight

Run training with:

```bash
cd HPE
python train.py --dataset ../Data/MM-Fi
```

Evaluation:

```bash
python eval2.py
```

### HAR: XRF55 Human Action Recognition

After data preprocessing, run training with:

```bash
cd HAR
python train.py
```

Evaluation example:

```bash
python eval_all.py --data_dir ../Data/Split_XRF55_Dataset --reload_path ./checkpoint/example/xrf55_last.pth
```

---

## Method Components

Core modules in this repository include:

### Meta-weighted fusion / weighting logic

- `HPE/task.py`
- `HAR/HAR_Task.py`

### Diffusion-based KD / alignment

- `HPE/meta_diffusion/losses/`
- `HAR/losses/`

### Task-specific backbones and heads

- `HPE/backbones/`
- `HAR/backbone_models/`

---



## Reproducibility Checklist

Before training, it is helpful to verify the following:

- MM-Fi or XRF55 is downloaded and placed under `Data/`
- pretrained backbones are in the expected subfolders
- `config.yaml` is checked for HPE experiments
- `split_train_test.py` has been run for HAR
- your PyTorch/CUDA environment matches your machine
- paths are correct relative to the script working directory

---



---

## Acknowledgement

This release preserves the original research code structure used during submission and supplementary-material preparation.

We also thank the authors of **X-Fi** for providing a clear reference implementation and setup pipeline that helps users prepare the datasets and pretrained backbones:

- https://github.com/xyanchen/X-Fi

---

## Contact

For questions about the code or paper, please open an issue in this repository.
