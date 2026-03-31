1. Setup
a. Download Data and Pretrained WeightsFirst, download the MM-Fi Dataset and the pretrained backbone model weights required by the framework.
b. Directory StructurePlace the downloaded MM-Fi dataset folder into a Data/ directory at the project root. The pretrained backbone models should be placed within the HPE/backbones/ directory, organized into subdirectories by modality.The final project structure should look like this:.
├── Data/
│   └── MM-Fi/
│       ├── P01/
│       └── ...
│
└── HPE/
    ├── backbones/
    │   ├── depth_benchmark/
    │   │   └── depth_Resnet18.pt
    │   ├── lidar_benchmark/
    │   │   └── lidar_all_random.pt
    │   └── CSI_benchmark/
    │       └── protocol3_random_1.pkl
    │
    ├── config.yaml      <- Configuration file
    ├── task.py          <- Midas model definition
    ├── train.py         <- Main training script
2. ConfigurationAll dataset and training parameters are controlled via the config.yaml file. Before running, please review and edit this file to match your experimental setup.Key parameters include:modality: Define the sensor combination to use (e.g., "depth|lidar|wifi-csi").protocol: Select the activity set from the dataset (e.g., 3 for all activities).split: Choose the predefined train/test split ('split_1', 'split_2', or 'split_3').train_loader.batch_size: Set the batch size for training.learning_rate: Set the learning rate for the main model parameters.meta_lr: Set the learning rate for the meta-parameters (modality weights).
3. TrainingTo train the Midas model for the HPE task, run the main training script. You must provide the path to the dataset directory as a command-line argument. The script will automatically load all other settings from config.yaml.# Example training command
python train.py --dataset ./Data/MM-Fi

