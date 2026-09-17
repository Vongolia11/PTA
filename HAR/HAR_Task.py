import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# NOTE:
# The original supplementary package referenced `NewTask`, but that file was
# omitted during later code reorganization. The public release uses the
# available Task implementation.
from Task import Task
from XRF55_Dataset import XRF55_Dataset
from DualNet import *
from Encoders import *
from engine import *
from Extractor import (
    mmwave_feature_extractor,
    wifi_feature_extractor,
    rfid_feature_extractor,
    classification_Head,
)
from misc import *
from backbone_models.mmWave.ResNet import *


class HAR_Task(Task):
    def __init__(self):
        super().__init__(
            task_name="xrf55_har",
            modalities=["mmwave", "wifi", "rfid"],
        )

    def set_encoders(self):
        mmwave_model = torch.load('./backbone_models/mmWave/mmwave_ResNet18.pt')
        mmwave_extractor = mmwave_feature_extractor(mmwave_model).eval()

        wifi_model = torch.load('./backbone_models/WIFI/wifi_ResNet18.pt')
        wifi_extractor = wifi_feature_extractor(wifi_model).eval()

        rfid_model = torch.load('./backbone_models/RFID/rfid_ResNet18.pt')
        rfid_extractor = rfid_feature_extractor(rfid_model).eval()

        # `.eval()` above does not freeze parameters. Task.train() subsequently
        # calls model.train(), and these encoder parameters remain in the main
        # optimizer. Thus the released HAR pipeline fine-tunes the encoders.
        encoders_list = [
            mmwave_extractor,
            wifi_extractor,
            rfid_extractor,
        ]

        encode_info = [
            (512, 512, 32),
            (512, 512, 4),
            (512, 512, 5),
        ]

        for i, name in enumerate(self.modalities):
            self.encoders[name] = Encoder(
                i,
                encoders_list[i],
                encode_info[i],
            )

    def set_decoder(self):
        self.decoder = classification_Head(
            emb_size=512,
            num_classes=55,
        )

    def set_losses(self):
        self.losses["ce"] = nn.CrossEntropyLoss()

    def set_train_data(self, val_split_ratio=0.2):
        full_dataset = XRF55_Dataset(
            root_dir="./Data/Split_XRF55_Dataset",
            is_train=True,
            split="train",
        )

        if not full_dataset:
            return None, None

        num_samples = len(full_dataset)
        val_size = int(val_split_ratio * num_samples)
        train_size = num_samples - val_size

        # Fixed split for the inner-loop / outer-loop subsets.
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn_padd,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn_padd,
        )

        return train_dataloader, val_dataloader

    def set_test_data(self, args):
        # Packaging fix: the released repository contains XRF55_Dataset rather
        # than the stale XRF55_Dataset_new identifier used in an older script.
        test_dataset = XRF55_Dataset(
            root_dir=args.data_dir,
            is_train=False,
            split="test",
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padd,
        )
        return test_dataloader


def collate_fn_padd(batch):
    """Collate XRF55 modality arrays and labels into tensors."""
    labels = []
    [labels.append(float(t[3])) for t in batch]
    labels = torch.FloatTensor(labels)

    mmwave_data = np.array([t[2] for t in batch])
    mmwave_data = torch.FloatTensor(mmwave_data)

    wifi_data = np.array([t[0] for t in batch])
    wifi_data = torch.FloatTensor(wifi_data)

    rfid_data = np.array([t[1] for t in batch])
    rfid_data = torch.FloatTensor(rfid_data)

    return mmwave_data, wifi_data, rfid_data, labels
