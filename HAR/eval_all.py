import torch
import torch.nn as nn
import argparse

from HAR_Task import HAR_Task
from utils import multi_test


def main():
    parser = argparse.ArgumentParser(description="Multi-modal HAR Evaluation Script")
    parser.add_argument('--data_dir', type=str,
                        default="./Data/Split_XRF55_Dataset",
                        help='path to dataset')
    parser.add_argument('--reload_path', type=str,
                        default="./HAR/checkpoint/example/",
                        help='path to pretrained model weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.reload_path, map_location=device)

    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        raise KeyError("Key 'model' not found in the checkpoint. Please check the saving method.")

    model.to(device)
    model.eval()

    har_task = HAR_Task()
    har_task.set_losses()

    test_loader = har_task.set_test_data(args)
    criterion = har_task.losses["ce"]
    multi_test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()

