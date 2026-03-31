import os
import shutil
from tqdm import tqdm


def split_scene1_data(split=14):
    root_ns = "./Data/XRF55_Dataset"
    dst_wr = "./Data/Split_XRF55_Dataset/"

    scene_to_process = "Scene1"

    print(f"Source directory: {root_ns}")
    print(f"Destination directory: {dst_wr}")
    print(f"Processing specified scene: {scene_to_process}")

    src_rfid_dir = os.path.join(root_ns, scene_to_process, 'RFID/')
    src_wifi_dir = os.path.join(root_ns, scene_to_process, 'WiFi/')
    src_mmwave_dir = os.path.join(root_ns, scene_to_process, 'mmWave/')

    dst_train_rfid = os.path.join(dst_wr, 'train_data', scene_to_process, 'RFID/')
    dst_train_wifi = os.path.join(dst_wr, 'train_data', scene_to_process, 'WiFi/')
    dst_train_mmwave = os.path.join(dst_wr, 'train_data', scene_to_process, 'mmWave/')

    dst_test_rfid = os.path.join(dst_wr, 'test_data', scene_to_process, 'RFID/')
    dst_test_wifi = os.path.join(dst_wr, 'test_data', scene_to_process, 'WiFi/')
    dst_test_mmwave = os.path.join(dst_wr, 'test_data', scene_to_process, 'mmWave/')

    os.makedirs(dst_train_rfid, exist_ok=True)
    os.makedirs(dst_train_wifi, exist_ok=True)
    os.makedirs(dst_train_mmwave, exist_ok=True)
    os.makedirs(dst_test_rfid, exist_ok=True)
    os.makedirs(dst_test_wifi, exist_ok=True)
    os.makedirs(dst_test_mmwave, exist_ok=True)

    if not os.path.exists(src_rfid_dir):
        print(f"Error: Directory '{scene_to_process}/RFID/' not found in '{root_ns}', please check the path.")
        return

    for file in tqdm(os.listdir(src_rfid_dir), desc=f"Processing {scene_to_process} files"):
        if not file.endswith('.npy'):
            continue

        try:
            actidx = int(file.split("_")[2].split(".")[0])
        except (IndexError, ValueError):
            print(f"\nWarning: Skipping file with incorrect format: {file}")
            continue

        if actidx <= split:
            shutil.copy(os.path.join(src_rfid_dir, file), os.path.join(dst_train_rfid, file))
            shutil.copy(os.path.join(src_wifi_dir, file), os.path.join(dst_train_wifi, file))
            shutil.copy(os.path.join(src_mmwave_dir, file), os.path.join(dst_train_mmwave, file))
        else:
            shutil.copy(os.path.join(src_rfid_dir, file), os.path.join(dst_test_rfid, file))
            shutil.copy(os.path.join(src_wifi_dir, file), os.path.join(dst_test_wifi, file))
            shutil.copy(os.path.join(src_mmwave_dir, file), os.path.join(dst_test_mmwave, file))


if __name__ == '__main__':
    split_scene1_data(split=14)

