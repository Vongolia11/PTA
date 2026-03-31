import torch
from torch import nn
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from syn_DI_dataset import make_dataset
from utils import collate_fn_padd
from evaluate import error
from task import Midas 

def train_one_epoch(model, train_loader, model_optimizer, criterion, device, lambda_kd):
    model.train()
    total_main_loss, total_task_loss, total_kd_loss = 0.0, 0.0, 0.0
    
    progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)
    for train_batch in progress_bar:
        rgb_data, depth_data, mmwave_data, lidar_data, csi_data, target, modality_list = train_batch
        
        corrected_modality_list = list(modality_list)
        if torch.all(rgb_data == 0): corrected_modality_list[0] = False
        if torch.all(depth_data == 0): corrected_modality_list[1] = False
        if torch.all(mmwave_data == 0): corrected_modality_list[2] = False
        if torch.all(lidar_data == 0): corrected_modality_list[3] = False
        if torch.all(csi_data == 0): corrected_modality_list[4] = False
        if not any(corrected_modality_list): continue
        
        datas = {
            'rgb': rgb_data.to(device), 'depth': depth_data.to(device), 'mmwave': mmwave_data.to(device),
            'lidar': lidar_data.to(device), 'csi': csi_data.to(device)
        }
        target = target.to(device)

        prediction, kd_loss = model(datas, corrected_modality_list, val=False)
        
        task_loss = criterion(prediction, target)
        main_loss = task_loss + lambda_kd * kd_loss
        
        model_optimizer.zero_grad()
        main_loss.backward()
        model_optimizer.step()
        
        total_main_loss += main_loss.item()
        total_task_loss += task_loss.item()
        total_kd_loss += kd_loss.item()
        
        progress_bar.set_postfix({"Task Loss": f"{task_loss.item():.4f}", "KD Loss": f"{kd_loss.item():.4f}"})

    avg_main_loss = total_main_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_task_loss = total_task_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_kd_loss = total_kd_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    return avg_main_loss, avg_task_loss, avg_kd_loss

def update_meta_weights(model, meta_val_loader, meta_optimizer, criterion, device):
    model.eval()
    for name, param in model.named_parameters():
        if 'kd_weights' not in name:
            param.requires_grad = False

    meta_optimizer.zero_grad()
    for val_batch in tqdm(meta_val_loader, desc="Updating Meta-Weights", leave=False):
        rgb_val, depth_val, mmwave_val, lidar_val, csi_val, target_val, modality_list_val = val_batch
        
        corrected_modality_list_val = list(modality_list_val)
        if torch.all(rgb_val == 0): corrected_modality_list_val[0] = False
        if not any(corrected_modality_list_val): continue

        datas = {
            'rgb': rgb_val.to(device), 'depth': depth_val.to(device), 'mmwave': mmwave_val.to(device),
            'lidar': lidar_val.to(device), 'csi': csi_val.to(device)
        }
        target_val = target_val.to(device)
        meta_pred, _ = model(datas, corrected_modality_list_val, val=True)
        meta_loss = criterion(meta_pred, target_val)
        meta_loss.backward()
    meta_optimizer.step()
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
def final_validate(model, val_loader, test_criterion, device):
    model.eval()
    total_mpjpe, total_pa_mpjpe, processed_batches = 0.0, 0.0, 0
    
    with torch.no_grad():
        for data_tuple in tqdm(val_loader, desc="Final Validation", leave=False):
            rgb_data, depth_data, mmwave_data, lidar_data, csi_data, target, modality_list = data_tuple

            corrected_modality_list = list(modality_list)
            if torch.all(rgb_data == 0): corrected_modality_list[0] = False
            if torch.all(depth_data == 0): corrected_modality_list[1] = False
            if torch.all(mmwave_data == 0): corrected_modality_list[2] = False
            if torch.all(lidar_data == 0): corrected_modality_list[3] = False
            if torch.all(csi_data == 0): corrected_modality_list[4] = False
            if not any(corrected_modality_list): continue

            datas = {
                'rgb': rgb_data.to(device), 'depth': depth_data.to(device), 'mmwave': mmwave_data.to(device),
                'lidar': lidar_data.to(device), 'csi': csi_data.to(device)
            }
            target_on_gpu = target.to(device)
            
            prediction_on_gpu, _ = model(datas, corrected_modality_list, val=True)
            
            prediction_numpy = prediction_on_gpu.cpu().numpy()
            target_numpy = target.cpu().numpy()
            
            mpjpe, pampjpe = test_criterion(prediction_numpy, target_numpy)
            
            total_mpjpe += mpjpe.item()
            total_pa_mpjpe += pampjpe.item()
            processed_batches += 1
            
    avg_mpjpe = total_mpjpe / processed_batches if processed_batches > 0 else float('inf')
    avg_pa_mpjpe = total_pa_mpjpe / processed_batches if processed_batches > 0 else float('inf')
    
    return avg_mpjpe, avg_pa_mpjpe

def main():
    parser = argparse.ArgumentParser('Midas model for MMFi HPE with MetaKD (Epoch-based)')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training from')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available training resources: {device}') 

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
        
    print("Preparing datasets for Meta-Learning...")
    full_train_dataset, val_dataset = make_dataset(args.dataset, config)
    
    n_train = len(full_train_dataset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * n_train))
    meta_train_idx, meta_val_idx = indices[:split], indices[split:]
    
    meta_train_dataset = Subset(full_train_dataset, meta_train_idx)
    meta_val_dataset = Subset(full_train_dataset, meta_val_idx)
    
    print(f"Full train set: {n_train} samples")
    print(f"Meta-train set: {len(meta_train_dataset)} samples")
    print(f"Meta-validation set: {len(meta_val_dataset)} samples")
    print(f"Final validation set: {len(val_dataset)} samples")

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    meta_train_loader = DataLoader(meta_train_dataset, **config['loader'], collate_fn=collate_fn_padd, generator=rng_generator)
    meta_val_loader = DataLoader(meta_val_dataset, **config['loader'], collate_fn=collate_fn_padd, generator=rng_generator)
    final_val_loader = DataLoader(val_dataset, **config['loader'], collate_fn=collate_fn_padd, generator=rng_generator)

    torch.manual_seed(3407)
    model = Midas()
    model.to(device)

    model_params = [p for name, p in model.named_parameters() if 'kd_weights' not in name]
    meta_params = [p for name, p in model.named_parameters() if 'kd_weights' in name]

    model_optimizer = torch.optim.Adam(model_params, lr=config['learning_rate'])
    meta_optimizer = torch.optim.Adam(meta_params, lr=config.get('meta_lr', 1e-2))

    train_criterion = nn.MSELoss()
    test_criterion = error
    
    num_epochs = config['training_epochs']
    lambda_kd = config.get('lambda_kd', 1.0)
    save_dir = './pre-trained_weights'
    os.makedirs(save_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_error = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_error = checkpoint['best_val_error']
            print(f"=> loaded checkpoint, resuming from epoch {start_epoch}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    print("Starting Meta-Training (Epoch-based)...")
    for epoch in range(start_epoch, num_epochs):
        
        avg_main_loss, avg_task_loss, avg_kd_loss = train_one_epoch(model, meta_train_loader, model_optimizer, train_criterion, device, lambda_kd)
        
    
        update_meta_weights(model, meta_val_loader, meta_optimizer, train_criterion, device)
        
        if hasattr(model, 'fusion_block') and hasattr(model.fusion_block, 'kd_weights'):
            print('Current kd_weights (raw):', model.fusion_block.kd_weights.data.cpu().numpy())
            normalized_weights = model.fusion_block.get_meta_weights()
            print('Current kd_weights (softmax):', normalized_weights.data.cpu().numpy())
        else:
            print('Warning: kd_weights not found in model. Please check model structure in task.py.')
        
        avg_mpjpe, avg_pa_mpjpe = final_validate(model, final_val_loader, test_criterion, device)
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} Finished ---")
        print(f"Avg Train Loss: {avg_main_loss:.4f} (Task: {avg_task_loss:.4f}, KD: {avg_kd_loss:.4f}), "
              f"Validation MPJPE: {avg_mpjpe*1000:.2f}mm, PA-MPJPE: {avg_pa_mpjpe*1000:.2f}mm")
        
        is_best = avg_mpjpe < best_val_error
        if is_best:
            best_val_error = avg_mpjpe
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path} with new best MPJPE: {best_val_error*1000:.2f}mm")

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_optimizer_state_dict': model_optimizer.state_dict(),
            'meta_optimizer_state_dict': meta_optimizer.state_dict(),
            'best_val_error': best_val_error,
        }, checkpoint_path)

    print("Training finished.")
    
if __name__ == '__main__':
    main()
