import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import itertools

from syn_DI_dataset import make_dataset
from utils import collate_fn_padd
from evaluate import error
from task import Midas

def generate_modality_combinations():
    modalities = [True, False]
    all_combinations = list(itertools.product(modalities, repeat=5))
    non_empty_combinations = [list(c) for c in all_combinations if any(c)]
    return non_empty_combinations

def evaluate_combination(model, val_loader, test_criterion, device, modality_combination, modality_names, limit_batches=None):
    model.eval()
    total_mpjpe, total_pa_mpjpe, processed_batches = 0.0, 0.0, 0
    
    combination_name = "+".join([name for name, active in zip(modality_names, modality_combination) if active])
    
    with torch.no_grad():
        for i, data_tuple in enumerate(tqdm(val_loader, desc=f"Evaluating {combination_name}", leave=False)):
            if limit_batches is not None and i >= limit_batches:
                break

            rgb_data, depth_data, mmwave_data, lidar_data, csi_data, target, _ = data_tuple
            
            if modality_combination[0] and torch.all(rgb_data == 0): continue
            if modality_combination[1] and torch.all(depth_data == 0): continue
            if modality_combination[2] and torch.all(mmwave_data == 0): continue
            if modality_combination[3] and torch.all(lidar_data == 0): continue
            if modality_combination[4] and torch.all(csi_data == 0): continue
            
            datas = {
                'rgb': rgb_data.to(device),
                'depth': depth_data.to(device),
                'mmwave': mmwave_data.to(device),
                'lidar': lidar_data.to(device),
                'csi': csi_data.to(device)
            }
            target_on_gpu = target.to(device)
            
            prediction_on_gpu, _ = model(datas, modality_combination, val=True)
            
            prediction_numpy = prediction_on_gpu.cpu().numpy()
            target_numpy = target.cpu().numpy()
            
            mpjpe, pampjpe = test_criterion(prediction_numpy, target_numpy)
            
            total_mpjpe += mpjpe.item()
            total_pa_mpjpe += pampjpe.item()
            processed_batches += 1
            
    avg_mpjpe = total_mpjpe / processed_batches if processed_batches > 0 else float('inf')
    avg_pa_mpjpe = total_pa_mpjpe / processed_batches if processed_batches > 0 else float('inf')
    
    return combination_name, avg_mpjpe, avg_pa_mpjpe

def main():
    parser = argparse.ArgumentParser(description='Midas Full Evaluation Script')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the saved model weights (e.g., best_model.pt)')
    parser.add_argument('--limit_batches', type=int, default=None, help='Limit the number of batches for a quick evaluation')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='Path to save the evaluation results')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}') 

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
        
    _, val_dataset = make_dataset(args.dataset, config)
    val_loader = DataLoader(val_dataset, **config['loader'], collate_fn=collate_fn_padd)

    model = Midas()
    
    if os.path.isfile(args.weights_path):
        print(f"=> loading weights from '{args.weights_path}'")
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    else:
        print(f"=> no weights found at '{args.weights_path}'")
        return
        
    model.to(device)
    model.eval()

    test_criterion = error
    modality_names = ["I", "D", "R", "L", "W"] 
    
    all_combinations = generate_modality_combinations()
    print("Filtering out combinations containing RGB (I) or mmWave (R)...")
    filtered_combinations = [
        combo for combo in all_combinations if not combo[0] and not combo[2]
    ]
    
    with open(args.output_file, 'w') as f:
        header1 = "--- Starting Filtered Evaluation ---"
        print(header1)
        f.write(header1 + '\n')
        
        if args.limit_batches:
            header2 = f"--- NOTE: Running in quick mode, evaluating on a maximum of {args.limit_batches} batches per combination. ---"
            print(header2)
            f.write(header2 + '\n')

        header3 = f"Found {len(filtered_combinations)} valid modality combinations to test."
        print(header3)
        f.write(header3 + '\n')

        separator = "--------------------------------------------------"
        print(separator)
        f.write(separator + '\n')

        table_header = f"{'Modality Combination':<20} | {'MPJPE (mm)':<15} | {'PA-MPJPE (mm)':<15}"
        print(table_header)
        f.write(table_header + '\n')

        print(separator)
        f.write(separator + '\n')
        for combo in filtered_combinations:
            combo_name, mpjpe, pa_mpjpe = evaluate_combination(model, val_loader, test_criterion, device, combo, modality_names, limit_batches=args.limit_batches)
            
            result_line = f"{combo_name:<20} | {mpjpe*1000:<15.2f} | {pa_mpjpe*1000:<15.2f}"
            print(result_line)
            f.write(result_line + '\n')
            
        print(separator)
        f.write(separator + '\n')
    
    print(f"Evaluation finished. Full results have been saved to '{args.output_file}'.")

if __name__ == '__main__':
    main()
