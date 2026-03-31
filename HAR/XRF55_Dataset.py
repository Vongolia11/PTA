import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob


class XRF55_Dataset(Dataset):
    def __init__(self, root_dir, is_train, split='train'):
        """
        Args:
            root_dir (string): Directory with all the data.
            is_train (bool): Deprecated, use 'split' instead. Kept for compatibility.
            split (string): 'train' or 'test'.
        """
        super(XRF55_Dataset, self).__init__()
        self.root_dir = root_dir

        if split == 'train':
            print("Loading training set ...")
            scenes_to_load = ["train_data/Scene1"]
        elif split == 'test':
            print("Loading testing set ...")
            scenes_to_load = ["test_data/Scene1"]

        self.path = self.root_dir
        self.RFID_name_list = []

        for scene_sub in scenes_to_load:
            scene_path = os.path.join(self.path, scene_sub, 'RFID')
            if not os.path.isdir(scene_path):
                print(f"Warning: Scene path not found {scene_path}")
                continue

            sub_list = glob.glob(os.path.join(scene_path, '*.npy'))
            sub_list.sort()
            self.RFID_name_list += sub_list

        print(f"Loading complete, found {len(self.RFID_name_list)} samples.")

    def __len__(self):
        return len(self.RFID_name_list)

    def __getitem__(self, idx):
        RFID_file_name = self.RFID_name_list[idx]

        parts = RFID_file_name.split(os.sep)
        base_path = os.sep.join(parts[:-2])
        file_basename = os.path.basename(RFID_file_name)

        WIFI_file_name = os.path.join(base_path, 'WiFi', file_basename)
        mmWave_file_name = os.path.join(base_path, 'mmWave', file_basename)

        try:
            wifi_data = self.load_wifi(WIFI_file_name)
            rfid_data = self.load_rfid(RFID_file_name)
            mmwave_data = self.load_mmwave(mmWave_file_name).reshape(17, 256, 128)
            label = int(os.path.basename(os.path.normpath(RFID_file_name)).split('_')[1]) - 1

            return wifi_data, rfid_data, mmwave_data, label
        except FileNotFoundError as e:
            print(f"Error: Could not find corresponding file when loading sample {idx}: {e}")
            return None, None, None, None

    def load_rfid(self, filename):
        return np.load(filename)

    def load_wifi(self, filename):
        return np.load(filename)

    def load_mmwave(self, filename):
        return np.load(filename)
