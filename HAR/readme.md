MIDAS
1. Data Setup
First, download the XRF55 Dataset.

Place the downloaded XRF55_Dataset folder into the Data/ directory in the project root. Download the pretrained encoder weights. The directory structure should look like this:

.
├── Data/
│   └── XRF55_Dataset/
│       ├── Scene1/
│       └── ...
│
└── HAR/
    └── backbone_models/
        ├── mmWave/
        │   └── mmwave_ResNet18.pt
        ├── Wi-Fi/
        │   └── wifi_ResNet18.pt
        └── RFID/
            └── rfid_ResNet18.pt

2. Prepare Data for Training
Next, run the splitting script to create the training and test sets:

python split_train_test.py

This script will process the raw data and create a new Split_XRF55_Dataset folder inside Data/ with the prepared data.

3. Training
To train the model, run the main training script:

python train.py