import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

def generate_none_empth_modality_list():
    """
    生成一个随机的模态可用性列表，但强制禁用RGB。
    """
    # 为 depth 和 mmwave 随机生成 True/False
    other_modalities = random.choices(
        [True, False],
        k=2,
        weights=[50, 50]
    )
    
    # 为 lidar 和 wifi-csi 设置较高的可用概率
    lidar_ = random.choices(
        [True, False],
        k=1,
        weights=[70, 30]
    )
    wifi_ = random.choices(
        [True, False],
        k=1,
        weights=[70, 30]
    )
    
    # 构造最终列表，将RGB（索引0）硬编码为False
    # 顺序: [RGB, Depth, mmWave, LiDAR, WiFi-CSI]
    final_list = [False] + [other_modalities[0]] + [other_modalities[1]] + lidar_ + wifi_
    
    # 确保至少有一个模态是可用的，防止列表全为False
    if sum(final_list) == 0:
        # 如果全为False，则重新生成一个
        return generate_none_empth_modality_list()
    else:
        return final_list
    
def collate_fn_padd(batch):
    '''
    Padds batch of variable length.
    
    这个版本会检查每个模态的数据是否存在，如果不存在则创建占位符，
    从而避免KeyError。
    '''
    batch_size = len(batch)
    
    # --- 关键点 ---
    # 检查第一个样本中存在哪些键，以确定哪些模态被加载了
    available_keys = batch[0].keys()

    # Ground truth keypoints
    kpts = []
    [kpts.append(np.array(t['output'])) for t in batch]
    kpts = torch.FloatTensor(np.array(kpts))

    # rgb
    if 'input_rgb' in available_keys:
        rgb_data = np.array([(t['input_rgb']) for t in batch ])
        rgb_data = torch.FloatTensor(rgb_data).permute(0,3,1,2)
    else:
        # 如果RGB数据不存在，创建一个占位符
        rgb_data = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32)

    # depth
    if 'input_depth' in available_keys:
        depth_data = np.array([(t['input_depth']) for t in batch ])
        depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)
    else:
        # 如果Depth数据不存在，创建一个占位符
        depth_data = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32)

    # mmwave
    if 'input_mmwave' in available_keys:
        mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch]
        mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data, batch_first=True)
    else:
        # 如果mmWave数据不存在，创建一个占位符
        mmwave_data = torch.zeros(batch_size, 1, 5, dtype=torch.float32) # (B, Seq, Feat)

    # lidar
    if 'input_lidar' in available_keys:
        lidar_data = [torch.Tensor(t['input_lidar']) for t in batch]
        lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data, batch_first=True)
    else:
        # 如果LiDAR数据不存在，创建一个占位符
        lidar_data = torch.zeros(batch_size, 1, 3, dtype=torch.float32) # (B, Seq, Feat)

    # wifi-csi
    if 'input_wifi-csi' in available_keys:
        wifi_data = np.array([(t['input_wifi-csi']) for t in batch])
        wifi_data = torch.FloatTensor(wifi_data)
    else:
        # 如果WiFi-CSI数据不存在，创建一个占位符
        wifi_data = torch.zeros(batch_size, 2, 114, 96, dtype=torch.float32)
    
    # 调用我们修复过的函数来生成模态列表
    modality_list = generate_none_empth_modality_list()

    return rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list

def hpe_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):
    model.eval()
    test_mpjpe = 0
    test_pampjpe = 0
    test_mse = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list = data
        rgb_data = rgb_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        kpts.to(device)
        labels = kpts.type(torch.FloatTensor)
        outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, modality_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        test_mse += criterion1(outputs,labels).item() * rgb_data.size(0)

        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        
        mpjpe, pampjpe = criterion2(outputs,labels)
        test_mpjpe += mpjpe.item() * rgb_data.size(0)
        test_pampjpe += pampjpe.item() * rgb_data.size(0)
    test_mpjpe = test_mpjpe/len(tensor_loader.dataset)
    test_pampjpe = test_pampjpe/len(tensor_loader.dataset)
    test_mse = test_mse/len(tensor_loader.dataset)
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(float(test_mse), float(test_mpjpe),float(test_pampjpe)))
    return test_mpjpe

def hpe_train(model, train_loader, test_loader, num_epochs, learning_rate, train_criterion, test_criterion, device, save_dir, val_random_seed):
    optimizer = torch.optim.AdamW(
        [
                {'params': model.linear_projector.parameters()},
                {'params': model.X_Fusion_block.parameters()}
            ],
        lr = learning_rate
    )
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_mpjpe = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        random.seed(epoch)
        num_iter = 1000
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list = data
                rgb_data = rgb_data.to(device)
                depth_data = depth_data.to(device)
                lidar_data = lidar_data.to(device)
                mmwave_data = mmwave_data.to(device)
                wifi_data = wifi_data.to(device)
                labels = kpts.to(device)
                labels = labels.type(torch.FloatTensor)
                
                optimizer.zero_grad()
                outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data,wifi_data, modality_list)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = train_criterion(outputs,labels)
                if loss == float('nan'):
                    print('nan')
                    print(outputs)
                    print(labels)
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * rgb_data.size(0)
            else:
                break
        epoch_loss = epoch_loss/(rgb_data.size(0)*num_iter)
        print('Epoch: {}, Loss: {:.8f}'.format(epoch, epoch_loss))
        if (epoch+1) % 10 == 0:
            test_mpjpe = hpe_test(
                model=model,
                tensor_loader=test_loader,
                criterion1 = train_criterion,
                criterion2 = test_criterion,
                device= device,
                val_random_seed = val_random_seed
            )
            if test_mpjpe <= best_test_mpjpe:
                print(f"best test mpjpe is:{test_mpjpe}")
                best_test_mpjpe = test_mpjpe
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):
    model.eval()
    rgb_test_mpjpe = 0
    rgb_test_pampjpe = 0
    rgb_test_mse = 0

    depth_test_mpjpe = 0
    depth_test_pampjpe = 0
    depth_test_mse = 0

    lidar_test_mpjpe = 0
    lidar_test_pampjpe = 0
    lidar_test_mse = 0

    mmwave_test_mpjpe = 0
    mmwave_test_pampjpe = 0
    mmwave_test_mse = 0

    wifi_test_mpjpe = 0
    wifi_test_pampjpe = 0
    wifi_test_mse = 0

    rgb_depth_test_mpjpe = 0
    rgb_depth_test_pampjpe = 0
    rgb_depth_test_mse = 0

    rgb_lidar_test_mpjpe = 0
    rgb_lidar_test_pampjpe = 0
    rgb_lidar_test_mse = 0

    rgb_mmwave_test_mpjpe = 0
    rgb_mmwave_test_pampjpe = 0
    rgb_mmwave_test_mse = 0

    rgb_wifi_test_mpjpe = 0
    rgb_wifi_test_pampjpe = 0
    rgb_wifi_test_mse = 0

    depth_lidar_test_mpjpe = 0
    depth_lidar_test_pampjpe = 0
    depth_lidar_test_mse = 0

    depth_mmwave_test_mpjpe = 0
    depth_mmwave_test_pampjpe = 0
    depth_mmwave_test_mse = 0

    depth_wifi_test_mpjpe = 0
    depth_wifi_test_pampjpe = 0
    depth_wifi_test_mse = 0

    lidar_mmwave_test_mpjpe = 0
    lidar_mmwave_test_pampjpe = 0
    lidar_mmwave_test_mse = 0

    lidar_wifi_test_mpjpe = 0
    lidar_wifi_test_pampjpe = 0
    lidar_wifi_test_mse = 0

    mmwave_wifi_test_mpjpe = 0
    mmwave_wifi_test_pampjpe = 0
    mmwave_wifi_test_mse = 0

    rgb_depth_lidar_test_mpjpe = 0
    rgb_depth_lidar_test_pampjpe = 0
    rgb_depth_lidar_test_mse = 0

    rgb_depth_mmwave_test_mpjpe = 0
    rgb_depth_mmwave_test_pampjpe = 0
    rgb_depth_mmwave_test_mse = 0

    rgb_depth_wifi_test_mpjpe = 0
    rgb_depth_wifi_test_pampjpe = 0
    rgb_depth_wifi_test_mse = 0

    rgb_lidar_mmwave_test_mpjpe = 0
    rgb_lidar_mmwave_test_pampjpe = 0
    rgb_lidar_mmwave_test_mse = 0

    rgb_lidar_wifi_test_mpjpe = 0
    rgb_lidar_wifi_test_pampjpe = 0
    rgb_lidar_wifi_test_mse = 0

    rgb_mmwave_wifi_test_mpjpe = 0
    rgb_mmwave_wifi_test_pampjpe = 0
    rgb_mmwave_wifi_test_mse = 0

    depth_lidar_mmwave_test_mpjpe = 0
    depth_lidar_mmwave_test_pampjpe = 0
    depth_lidar_mmwave_test_mse = 0

    depth_lidar_wifi_test_mpjpe = 0
    depth_lidar_wifi_test_pampjpe = 0
    depth_lidar_wifi_test_mse = 0

    depth_mmwave_wifi_test_mpjpe = 0
    depth_mmwave_wifi_test_pampjpe = 0
    depth_mmwave_wifi_test_mse = 0

    lidar_mmwave_wifi_test_mpjpe = 0
    lidar_mmwave_wifi_test_pampjpe = 0
    lidar_mmwave_wifi_test_mse = 0

    rgb_depth_lidar_mmwave_test_mpjpe = 0
    rgb_depth_lidar_mmwave_test_pampjpe = 0
    rgb_depth_lidar_mmwave_test_mse = 0

    rgb_depth_lidar_wifi_test_mpjpe = 0
    rgb_depth_lidar_wifi_test_pampjpe = 0
    rgb_depth_lidar_wifi_test_mse = 0

    rgb_depth_mmwave_wifi_test_mpjpe = 0
    rgb_depth_mmwave_wifi_test_pampjpe = 0
    rgb_depth_mmwave_wifi_test_mse = 0

    rgb_lidar_mmwave_wifi_test_mpjpe = 0
    rgb_lidar_mmwave_wifi_test_pampjpe = 0
    rgb_lidar_mmwave_wifi_test_mse = 0

    depth_lidar_mmwave_wifi_test_mpjpe = 0
    depth_lidar_mmwave_wifi_test_pampjpe = 0
    depth_lidar_mmwave_wifi_test_mse = 0

    rgb_depth_lidar_mmwave_wifi_test_mpjpe = 0
    rgb_depth_lidar_mmwave_wifi_test_pampjpe = 0
    rgb_depth_lidar_mmwave_wifi_test_mse = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, _ = data
        rgb_data = rgb_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        kpts.to(device)
        labels = kpts.type(torch.FloatTensor)
        labels_ = labels.detach().numpy()


        ' SINGLE MODALITY '
        ### rgb
        rgb_modality_list = [True, False, False, False, False]
        rgb_outputs = model(rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, rgb_modality_list)
        rgb_outputs = rgb_outputs.type(torch.FloatTensor)
        rgb_outputs.to(device)
        rgb_test_mse += criterion1(rgb_outputs,labels).item() * rgb_data.size(0)
        rgb_outputs = rgb_outputs.detach().numpy()
        rgb_mpjpe, rgb_pampjpe = criterion2(rgb_outputs,labels_)
        rgb_test_mpjpe += rgb_mpjpe.item() * rgb_data.size(0)
        rgb_test_pampjpe += rgb_pampjpe.item() * rgb_data.size(0)
        ### depth
        depth_modality_list = [False, True, False, False, False]
        depth_outputs = model(rgb_data, depth_data, mmwave_data, lidar_data, wifi_data, depth_modality_list)
        depth_outputs = depth_outputs.type(torch.FloatTensor)
        depth_outputs.to(device)
        depth_test_mse += criterion1(depth_outputs,labels).item() * rgb_data.size(0)
        depth_outputs = depth_outputs.detach().numpy()
        depth_mpjpe, depth_pampjpe = criterion2(depth_outputs,labels_)
        depth_test_mpjpe += depth_mpjpe.item() * rgb_data.size(0)
        depth_test_pampjpe += depth_pampjpe.item() * rgb_data.size(0)
        ### lidar
        lidar_modality_list = [False, False, False, True, False]
        lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_modality_list)
        lidar_outputs = lidar_outputs.type(torch.FloatTensor)
        lidar_outputs.to(device)
        lidar_test_mse += criterion1(lidar_outputs,labels).item() * rgb_data.size(0)
        lidar_outputs = lidar_outputs.detach().numpy()
        lidar_mpjpe, lidar_pampjpe = criterion2(lidar_outputs,labels_)
        lidar_test_mpjpe += lidar_mpjpe.item() * rgb_data.size(0)
        lidar_test_pampjpe += lidar_pampjpe.item() * rgb_data.size(0)
        ### mmwave
        mmwave_modality_list = [False, False, True, False, False]
        mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, mmwave_modality_list)
        mmwave_outputs = mmwave_outputs.type(torch.FloatTensor)
        mmwave_outputs.to(device)
        mmwave_test_mse += criterion1(mmwave_outputs,labels).item() * rgb_data.size(0)
        mmwave_outputs = mmwave_outputs.detach().numpy()
        mmwave_mpjpe, mmwave_pampjpe = criterion2(mmwave_outputs,labels_)
        mmwave_test_mpjpe += mmwave_mpjpe.item() * rgb_data.size(0)
        mmwave_test_pampjpe += mmwave_pampjpe.item() * rgb_data.size(0)
        ### wifi-cis
        wifi_modality_list = [False, False, False, False, True]
        wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, wifi_modality_list)
        wifi_outputs = wifi_outputs.type(torch.FloatTensor)
        wifi_outputs.to(device)
        wifi_test_mse += criterion1(wifi_outputs,labels).item() * rgb_data.size(0)
        wifi_outputs = wifi_outputs.detach().numpy()
        wifi_mpjpe, wifi_pampjpe = criterion2(wifi_outputs,labels_)
        wifi_test_mpjpe += wifi_mpjpe.item() * rgb_data.size(0)
        wifi_test_pampjpe += wifi_pampjpe.item() * rgb_data.size(0)
        
        'Dual modality'
        ### rgb + depth
        rgb_depth_modality_list = [True, True, False, False, False]
        rgb_depth_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_modality_list)
        rgb_depth_outputs = rgb_depth_outputs.type(torch.FloatTensor)
        rgb_depth_outputs.to(device)
        rgb_depth_test_mse += criterion1(rgb_depth_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_outputs = rgb_depth_outputs.detach().numpy()
        rgb_depth_mpjpe, rgb_depth_pampjpe = criterion2(rgb_depth_outputs,labels_)
        rgb_depth_test_mpjpe += rgb_depth_mpjpe.item() * rgb_data.size(0)
        rgb_depth_test_pampjpe += rgb_depth_pampjpe.item() * rgb_data.size(0)
        ### rgb + lidar
        rgb_lidar_modality_list = [True, False, False, True, False]
        rgb_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_lidar_modality_list)
        rgb_lidar_outputs = rgb_lidar_outputs.type(torch.FloatTensor)
        rgb_lidar_outputs.to(device)
        rgb_lidar_test_mse += criterion1(rgb_lidar_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_outputs = rgb_lidar_outputs.detach().numpy()
        rgb_lidar_mpjpe, rgb_lidar_pampjpe = criterion2(rgb_lidar_outputs,labels_)
        rgb_lidar_test_mpjpe += rgb_lidar_mpjpe.item() * rgb_data.size(0)
        rgb_lidar_test_pampjpe += rgb_lidar_pampjpe.item() * rgb_data.size(0)
        ### rgb + mmwave
        rgb_mmwave_modality_list = [True, False, True, False, False]
        rgb_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_mmwave_modality_list)
        rgb_mmwave_outputs = rgb_mmwave_outputs.type(torch.FloatTensor)
        rgb_mmwave_outputs.to(device)
        rgb_mmwave_test_mse += criterion1(rgb_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_mmwave_outputs = rgb_mmwave_outputs.detach().numpy()
        rgb_mmwave_mpjpe, rgb_mmwave_pampjpe = criterion2(rgb_mmwave_outputs,labels_)
        rgb_mmwave_test_mpjpe += rgb_mmwave_mpjpe.item() * rgb_data.size(0)
        rgb_mmwave_test_pampjpe += rgb_mmwave_pampjpe.item() * rgb_data.size(0)
        ### rgb + wifi
        rgb_wifi_modality_list = [True, False, False, False, True]
        rgb_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_wifi_modality_list)
        rgb_wifi_outputs = rgb_wifi_outputs.type(torch.FloatTensor)
        rgb_wifi_outputs.to(device)
        rgb_wifi_test_mse += criterion1(rgb_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_wifi_outputs = rgb_wifi_outputs.detach().numpy()
        rgb_wifi_mpjpe, rgb_wifi_pampjpe = criterion2(rgb_wifi_outputs,labels_)
        rgb_wifi_test_mpjpe += rgb_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_wifi_test_pampjpe += rgb_wifi_pampjpe.item() * rgb_data.size(0)
        ### depth + lidar
        depth_lidar_modality_list = [False, True, False, True, False]
        depth_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_modality_list)
        depth_lidar_outputs = depth_lidar_outputs.type(torch.FloatTensor)
        depth_lidar_outputs.to(device)
        depth_lidar_test_mse += criterion1(depth_lidar_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_outputs = depth_lidar_outputs.detach().numpy()
        depth_lidar_mpjpe, depth_lidar_pampjpe = criterion2(depth_lidar_outputs,labels_)
        depth_lidar_test_mpjpe += depth_lidar_mpjpe.item() * rgb_data.size(0)
        depth_lidar_test_pampjpe += depth_lidar_pampjpe.item() * rgb_data.size(0)
        ### depth + mmwave
        depth_mmwave_modality_list = [False, True, True, False, False]
        depth_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_mmwave_modality_list)
        depth_mmwave_outputs = depth_mmwave_outputs.type(torch.FloatTensor)
        depth_mmwave_outputs.to(device)
        depth_mmwave_test_mse += criterion1(depth_mmwave_outputs,labels).item() * rgb_data.size(0)
        depth_mmwave_outputs = depth_mmwave_outputs.detach().numpy()
        depth_mmwave_mpjpe, depth_mmwave_pampjpe = criterion2(depth_mmwave_outputs,labels_)
        depth_mmwave_test_mpjpe += depth_mmwave_mpjpe.item() * rgb_data.size(0)
        depth_mmwave_test_pampjpe += depth_mmwave_pampjpe.item() * rgb_data.size(0)
        ### depth + wifi
        depth_wifi_modality_list = [False, True, False, False, True]
        depth_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_wifi_modality_list)
        depth_wifi_outputs = depth_wifi_outputs.type(torch.FloatTensor)
        depth_wifi_outputs.to(device)
        depth_wifi_test_mse += criterion1(depth_wifi_outputs,labels).item() * rgb_data.size(0)
        depth_wifi_outputs = depth_wifi_outputs.detach().numpy()
        depth_wifi_mpjpe, depth_wifi_pampjpe = criterion2(depth_wifi_outputs,labels_)
        depth_wifi_test_mpjpe += depth_wifi_mpjpe.item() * rgb_data.size(0)
        depth_wifi_test_pampjpe += depth_wifi_pampjpe.item() * rgb_data.size(0)
        ### lidar + mmwave
        lidar_mmwave_modality_list = [False, False, True, True, False]
        lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_mmwave_modality_list)
        lidar_mmwave_outputs = lidar_mmwave_outputs.type(torch.FloatTensor)
        lidar_mmwave_outputs.to(device)
        lidar_mmwave_test_mse += criterion1(lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        lidar_mmwave_outputs = lidar_mmwave_outputs.detach().numpy()
        lidar_mmwave_mpjpe, lidar_mmwave_pampjpe = criterion2(lidar_mmwave_outputs,labels_)
        lidar_mmwave_test_mpjpe += lidar_mmwave_mpjpe.item() * rgb_data.size(0)
        lidar_mmwave_test_pampjpe += lidar_mmwave_pampjpe.item() * rgb_data.size(0)
        ### lidar + wifi
        lidar_wifi_modality_list = [False, False, False, True, True]
        lidar_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_wifi_modality_list)
        lidar_wifi_outputs = lidar_wifi_outputs.type(torch.FloatTensor)
        lidar_wifi_outputs.to(device)
        lidar_wifi_test_mse += criterion1(lidar_wifi_outputs,labels).item() * rgb_data.size(0)
        lidar_wifi_outputs = lidar_wifi_outputs.detach().numpy()
        lidar_wifi_mpjpe, lidar_wifi_pampjpe = criterion2(lidar_wifi_outputs,labels_)
        lidar_wifi_test_mpjpe += lidar_wifi_mpjpe.item() * rgb_data.size(0)
        lidar_wifi_test_pampjpe += lidar_wifi_pampjpe.item() * rgb_data.size(0)
        ### mmwave + wifi
        mmwave_wifi_modality_list = [False, False, True, False, True]
        mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, mmwave_wifi_modality_list)
        mmwave_wifi_outputs = mmwave_wifi_outputs.type(torch.FloatTensor)
        mmwave_wifi_outputs.to(device)
        mmwave_wifi_test_mse += criterion1(mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        mmwave_wifi_outputs = mmwave_wifi_outputs.detach().numpy()
        mmwave_wifi_mpjpe, mmwave_wifi_pampjpe = criterion2(mmwave_wifi_outputs,labels_)
        mmwave_wifi_test_mpjpe += mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        mmwave_wifi_test_pampjpe += mmwave_wifi_pampjpe.item() * rgb_data.size(0)

        'Three modality'
        ### rgb + depth + lidar
        rgb_depth_lidar_modality_list = [True, True, False, True, False]
        rgb_depth_lidar_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_lidar_modality_list)
        rgb_depth_lidar_outputs = rgb_depth_lidar_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_outputs.to(device)
        rgb_depth_lidar_test_mse += criterion1(rgb_depth_lidar_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_outputs = rgb_depth_lidar_outputs.detach().numpy()
        rgb_depth_lidar_mpjpe, rgb_depth_lidar_pampjpe = criterion2(rgb_depth_lidar_outputs,labels_)
        rgb_depth_lidar_test_mpjpe += rgb_depth_lidar_mpjpe.item() * rgb_data.size(0)
        rgb_depth_lidar_test_pampjpe += rgb_depth_lidar_pampjpe.item() * rgb_data.size(0)
        ### rgb + depth + mmwave
        rgb_depth_mmwave_modality_list = [True, True, True, False, False]
        rgb_depth_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_mmwave_modality_list)
        rgb_depth_mmwave_outputs = rgb_depth_mmwave_outputs.type(torch.FloatTensor)
        rgb_depth_mmwave_outputs.to(device)
        rgb_depth_mmwave_test_mse += criterion1(rgb_depth_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_mmwave_outputs = rgb_depth_mmwave_outputs.detach().numpy()
        rgb_depth_mmwave_mpjpe, rgb_depth_mmwave_pampjpe = criterion2(rgb_depth_mmwave_outputs,labels_)
        rgb_depth_mmwave_test_mpjpe += rgb_depth_mmwave_mpjpe.item() * rgb_data.size(0)
        rgb_depth_mmwave_test_pampjpe += rgb_depth_mmwave_pampjpe.item() * rgb_data.size(0)
        ### rgb + depth + wifi
        rgb_depth_wifi_modality_list = [True, True, False, False, True]
        rgb_depth_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_wifi_modality_list)
        rgb_depth_wifi_outputs = rgb_depth_wifi_outputs.type(torch.FloatTensor)
        rgb_depth_wifi_outputs.to(device)
        rgb_depth_wifi_test_mse += criterion1(rgb_depth_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_wifi_outputs = rgb_depth_wifi_outputs.detach().numpy()
        rgb_depth_wifi_mpjpe, rgb_depth_wifi_pampjpe = criterion2(rgb_depth_wifi_outputs,labels_)
        rgb_depth_wifi_test_mpjpe += rgb_depth_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_depth_wifi_test_pampjpe += rgb_depth_wifi_pampjpe.item() * rgb_data.size(0)
        ### rgb + lidar + mmwave
        rgb_lidar_mmwave_modality_list = [True, False, True, True, False]
        rgb_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_lidar_mmwave_modality_list)
        rgb_lidar_mmwave_outputs = rgb_lidar_mmwave_outputs.type(torch.FloatTensor)
        rgb_lidar_mmwave_outputs.to(device)
        rgb_lidar_mmwave_test_mse += criterion1(rgb_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_mmwave_outputs = rgb_lidar_mmwave_outputs.detach().numpy()
        rgb_lidar_mmwave_mpjpe, rgb_lidar_mmwave_pampjpe = criterion2(rgb_lidar_mmwave_outputs,labels_)
        rgb_lidar_mmwave_test_mpjpe += rgb_lidar_mmwave_mpjpe.item() * rgb_data.size(0)
        rgb_lidar_mmwave_test_pampjpe += rgb_lidar_mmwave_pampjpe.item() * rgb_data.size(0)
        ### rgb + lidar + wifi
        rgb_lidar_wifi_modality_list = [True, False, False, True, True]
        rgb_lidar_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_lidar_wifi_modality_list)
        rgb_lidar_wifi_outputs = rgb_lidar_wifi_outputs.type(torch.FloatTensor)
        rgb_lidar_wifi_outputs.to(device)
        rgb_lidar_wifi_test_mse += criterion1(rgb_lidar_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_wifi_outputs = rgb_lidar_wifi_outputs.detach().numpy()
        rgb_lidar_wifi_mpjpe, rgb_lidar_wifi_pampjpe = criterion2(rgb_lidar_wifi_outputs,labels_)
        rgb_lidar_wifi_test_mpjpe += rgb_lidar_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_lidar_wifi_test_pampjpe += rgb_lidar_wifi_pampjpe.item() * rgb_data.size(0)
        ### rgb + mmwave + wifi
        rgb_mmwave_wifi_modality_list = [True, False, True, False, True]
        rgb_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_mmwave_wifi_modality_list)
        rgb_mmwave_wifi_outputs = rgb_mmwave_wifi_outputs.type(torch.FloatTensor)
        rgb_mmwave_wifi_outputs.to(device)
        rgb_mmwave_wifi_test_mse += criterion1(rgb_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_mmwave_wifi_outputs = rgb_mmwave_wifi_outputs.detach().numpy()
        rgb_mmwave_wifi_mpjpe, rgb_mmwave_wifi_pampjpe = criterion2(rgb_mmwave_wifi_outputs,labels_)
        rgb_mmwave_wifi_test_mpjpe += rgb_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_mmwave_wifi_test_pampjpe += rgb_mmwave_wifi_pampjpe.item() * rgb_data.size(0)
        ### depth + lidar + mmwave
        depth_lidar_mmwave_modality_list = [False, True, True, True, False]
        depth_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_mmwave_modality_list)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_outputs.to(device)
        depth_lidar_mmwave_test_mse += criterion1(depth_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.detach().numpy()
        depth_lidar_mmwave_mpjpe, depth_lidar_mmwave_pampjpe = criterion2(depth_lidar_mmwave_outputs,labels_)
        depth_lidar_mmwave_test_mpjpe += depth_lidar_mmwave_mpjpe.item() * rgb_data.size(0)
        depth_lidar_mmwave_test_pampjpe += depth_lidar_mmwave_pampjpe.item() * rgb_data.size(0)
        ### depth + lidar + wifi
        depth_lidar_wifi_modality_list = [False, True, False, True, True]
        depth_lidar_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_wifi_modality_list)
        depth_lidar_wifi_outputs = depth_lidar_wifi_outputs.type(torch.FloatTensor)
        depth_lidar_wifi_outputs.to(device)
        depth_lidar_wifi_test_mse += criterion1(depth_lidar_wifi_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_wifi_outputs = depth_lidar_wifi_outputs.detach().numpy()
        depth_lidar_wifi_mpjpe, depth_lidar_wifi_pampjpe = criterion2(depth_lidar_wifi_outputs,labels_)
        depth_lidar_wifi_test_mpjpe += depth_lidar_wifi_mpjpe.item() * rgb_data.size(0)
        depth_lidar_wifi_test_pampjpe += depth_lidar_wifi_pampjpe.item() * rgb_data.size(0)
        ### depth + mmwave + wifi
        depth_mmwave_wifi_modality_list = [False, True, True, False, True]
        depth_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_mmwave_wifi_modality_list)
        depth_mmwave_wifi_outputs = depth_mmwave_wifi_outputs.type(torch.FloatTensor)
        depth_mmwave_wifi_outputs.to(device)
        depth_mmwave_wifi_test_mse += criterion1(depth_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        depth_mmwave_wifi_outputs = depth_mmwave_wifi_outputs.detach().numpy()
        depth_mmwave_wifi_mpjpe, depth_mmwave_wifi_pampjpe = criterion2(depth_mmwave_wifi_outputs,labels_)
        depth_mmwave_wifi_test_mpjpe += depth_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        depth_mmwave_wifi_test_pampjpe += depth_mmwave_wifi_pampjpe.item() * rgb_data.size(0)
        ### lidar + mmwave + wifi
        lidar_mmwave_wifi_modality_list = [False, False, True, True, True]
        lidar_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_mmwave_wifi_modality_list)
        lidar_mmwave_wifi_outputs = lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        lidar_mmwave_wifi_outputs.to(device)
        lidar_mmwave_wifi_test_mse += criterion1(lidar_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        lidar_mmwave_wifi_outputs = lidar_mmwave_wifi_outputs.detach().numpy()
        lidar_mmwave_wifi_mpjpe, lidar_mmwave_wifi_pampjpe = criterion2(lidar_mmwave_wifi_outputs,labels_)
        lidar_mmwave_wifi_test_mpjpe += lidar_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        lidar_mmwave_wifi_test_pampjpe += lidar_mmwave_wifi_pampjpe.item() * rgb_data.size(0)

        'Four modality'
        ### rgb + depth + lidar + mmwave
        rgb_depth_lidar_mmwave_modality_list = [True, True, True, True, False]
        rgb_depth_lidar_mmwave_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_lidar_mmwave_modality_list)
        rgb_depth_lidar_mmwave_outputs = rgb_depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_mmwave_outputs.to(device)
        rgb_depth_lidar_mmwave_test_mse += criterion1(rgb_depth_lidar_mmwave_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_mmwave_outputs = rgb_depth_lidar_mmwave_outputs.detach().numpy()
        rgb_depth_lidar_mmwave_mpjpe, rgb_depth_lidar_mmwave_pampjpe = criterion2(rgb_depth_lidar_mmwave_outputs,labels_)
        rgb_depth_lidar_mmwave_test_mpjpe += rgb_depth_lidar_mmwave_mpjpe.item() * rgb_data.size(0)
        rgb_depth_lidar_mmwave_test_pampjpe += rgb_depth_lidar_mmwave_pampjpe.item() * rgb_data.size(0)
        ### rgb + depth + lidar + wifi
        rgb_depth_lidar_wifi_modality_list = [True, True, False, True, True]
        rgb_depth_lidar_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_lidar_wifi_modality_list)
        rgb_depth_lidar_wifi_outputs = rgb_depth_lidar_wifi_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_wifi_outputs.to(device)
        rgb_depth_lidar_wifi_test_mse += criterion1(rgb_depth_lidar_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_wifi_outputs = rgb_depth_lidar_wifi_outputs.detach().numpy()
        rgb_depth_lidar_wifi_mpjpe, rgb_depth_lidar_wifi_pampjpe = criterion2(rgb_depth_lidar_wifi_outputs,labels_)
        rgb_depth_lidar_wifi_test_mpjpe += rgb_depth_lidar_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_depth_lidar_wifi_test_pampjpe += rgb_depth_lidar_wifi_pampjpe.item() * rgb_data.size(0)
        ### rgb + depth + mmwave + wifi
        rgb_depth_mmwave_wifi_modality_list = [True, True, True, False, True]
        rgb_depth_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_mmwave_wifi_modality_list)
        rgb_depth_mmwave_wifi_outputs = rgb_depth_mmwave_wifi_outputs.type(torch.FloatTensor)
        rgb_depth_mmwave_wifi_outputs.to(device)
        rgb_depth_mmwave_wifi_test_mse += criterion1(rgb_depth_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_mmwave_wifi_outputs = rgb_depth_mmwave_wifi_outputs.detach().numpy()
        rgb_depth_mmwave_wifi_mpjpe, rgb_depth_mmwave_wifi_pampjpe = criterion2(rgb_depth_mmwave_wifi_outputs,labels_)
        rgb_depth_mmwave_wifi_test_mpjpe += rgb_depth_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_depth_mmwave_wifi_test_pampjpe += rgb_depth_mmwave_wifi_pampjpe.item() * rgb_data.size(0)
        ### rgb + lidar + mmwave + wifi
        rgb_lidar_mmwave_wifi_modality_list = [True, False, True, True, True]
        rgb_lidar_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_lidar_mmwave_wifi_modality_list)
        rgb_lidar_mmwave_wifi_outputs = rgb_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        rgb_lidar_mmwave_wifi_outputs.to(device)
        rgb_lidar_mmwave_wifi_test_mse += criterion1(rgb_lidar_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_lidar_mmwave_wifi_outputs = rgb_lidar_mmwave_wifi_outputs.detach().numpy()
        rgb_lidar_mmwave_wifi_mpjpe, rgb_lidar_mmwave_wifi_pampjpe = criterion2(rgb_lidar_mmwave_wifi_outputs,labels_)
        rgb_lidar_mmwave_wifi_test_mpjpe += rgb_lidar_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_lidar_mmwave_wifi_test_pampjpe += rgb_lidar_mmwave_wifi_pampjpe.item() * rgb_data.size(0)
        ### depth + lidar + mmwave + wifi
        depth_lidar_mmwave_wifi_modality_list = [False, True, True, True, True]
        depth_lidar_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_mmwave_wifi_modality_list)
        depth_lidar_mmwave_wifi_outputs = depth_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_wifi_outputs.to(device)
        depth_lidar_mmwave_wifi_test_mse += criterion1(depth_lidar_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        depth_lidar_mmwave_wifi_outputs = depth_lidar_mmwave_wifi_outputs.detach().numpy()
        depth_lidar_mmwave_wifi_mpjpe, depth_lidar_mmwave_wifi_pampjpe = criterion2(depth_lidar_mmwave_wifi_outputs,labels_)
        depth_lidar_mmwave_wifi_test_mpjpe += depth_lidar_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        depth_lidar_mmwave_wifi_test_pampjpe += depth_lidar_mmwave_wifi_pampjpe.item() * rgb_data.size(0)

        'ALL modality'
        ### rgb + depth + lidar + mmwave + wifi
        rgb_depth_lidar_mmwave_wifi_modality_list = [True, True, True, True, True]
        rgb_depth_lidar_mmwave_wifi_outputs = model(rgb_data, depth_data,  mmwave_data, lidar_data, wifi_data, rgb_depth_lidar_mmwave_wifi_modality_list)
        rgb_depth_lidar_mmwave_wifi_outputs = rgb_depth_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        rgb_depth_lidar_mmwave_wifi_outputs.to(device)
        rgb_depth_lidar_mmwave_wifi_test_mse += criterion1(rgb_depth_lidar_mmwave_wifi_outputs,labels).item() * rgb_data.size(0)
        rgb_depth_lidar_mmwave_wifi_outputs = rgb_depth_lidar_mmwave_wifi_outputs.detach().numpy()
        rgb_depth_lidar_mmwave_wifi_mpjpe, rgb_depth_lidar_mmwave_wifi_pampjpe = criterion2(rgb_depth_lidar_mmwave_wifi_outputs,labels_)
        rgb_depth_lidar_mmwave_wifi_test_mpjpe += rgb_depth_lidar_mmwave_wifi_mpjpe.item() * rgb_data.size(0)
        rgb_depth_lidar_mmwave_wifi_test_pampjpe += rgb_depth_lidar_mmwave_wifi_pampjpe.item() * rgb_data.size(0)


    'single modality'
    ### rgb
    rgb_test_mpjpe = rgb_test_mpjpe/len(tensor_loader.dataset)
    rgb_test_pampjpe = rgb_test_pampjpe/len(tensor_loader.dataset)
    rgb_test_mse = rgb_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB',float(rgb_test_mse), float(rgb_test_mpjpe),float(rgb_test_pampjpe)))
    ### depth
    depth_test_mpjpe = depth_test_mpjpe/len(tensor_loader.dataset)
    depth_test_pampjpe = depth_test_pampjpe/len(tensor_loader.dataset)
    depth_test_mse = depth_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth',float(depth_test_mse), float(depth_test_mpjpe),float(depth_test_pampjpe)))
    ### lidar
    lidar_test_mpjpe = lidar_test_mpjpe/len(tensor_loader.dataset)
    lidar_test_pampjpe = lidar_test_pampjpe/len(tensor_loader.dataset)
    lidar_test_mse = lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar',float(lidar_test_mse), float(lidar_test_mpjpe),float(lidar_test_pampjpe)))
    ### mmwave
    mmwave_test_mpjpe = mmwave_test_mpjpe/len(tensor_loader.dataset)
    mmwave_test_pampjpe = mmwave_test_pampjpe/len(tensor_loader.dataset)
    mmwave_test_mse = mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('mmWave',float(mmwave_test_mse), float(mmwave_test_mpjpe),float(mmwave_test_pampjpe)))
    ### wifi
    wifi_test_mpjpe = wifi_test_mpjpe/len(tensor_loader.dataset)
    wifi_test_pampjpe = wifi_test_pampjpe/len(tensor_loader.dataset)
    wifi_test_mse = wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('WiFi-CSI',float(wifi_test_mse), float(wifi_test_mpjpe),float(wifi_test_pampjpe)))
    
    'dual modality'
    ### rgb + depth
    rgb_depth_test_mpjpe = rgb_depth_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_test_pampjpe = rgb_depth_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_test_mse = rgb_depth_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+Depth',float(rgb_depth_test_mse), float(rgb_depth_test_mpjpe),float(rgb_depth_test_pampjpe)))
    ### rgb + lidar
    rgb_lidar_test_mpjpe = rgb_lidar_test_mpjpe/len(tensor_loader.dataset)
    rgb_lidar_test_pampjpe = rgb_lidar_test_pampjpe/len(tensor_loader.dataset)
    rgb_lidar_test_mse = rgb_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+Lidar',float(rgb_lidar_test_mse), float(rgb_lidar_test_mpjpe),float(rgb_lidar_test_pampjpe)))
    ### rgb + mmwave
    rgb_mmwave_test_mpjpe = rgb_mmwave_test_mpjpe/len(tensor_loader.dataset)
    rgb_mmwave_test_pampjpe = rgb_mmwave_test_pampjpe/len(tensor_loader.dataset)
    rgb_mmwave_test_mse = rgb_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+mmWave',float(rgb_mmwave_test_mse), float(rgb_mmwave_test_mpjpe),float(rgb_mmwave_test_pampjpe)))
    ### rgb + wifi
    rgb_wifi_test_mpjpe = rgb_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_wifi_test_pampjpe = rgb_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_wifi_test_mse = rgb_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+WiFi-CSI',float(rgb_wifi_test_mse), float(rgb_wifi_test_mpjpe),float(rgb_wifi_test_pampjpe)))
    ### depth + lidar
    depth_lidar_test_mpjpe = depth_lidar_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_test_pampjpe = depth_lidar_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_test_mse = depth_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar',float(depth_lidar_test_mse), float(depth_lidar_test_mpjpe),float(depth_lidar_test_pampjpe)))
    ### depth + mmwave
    depth_mmwave_test_mpjpe = depth_mmwave_test_mpjpe/len(tensor_loader.dataset)
    depth_mmwave_test_pampjpe = depth_mmwave_test_pampjpe/len(tensor_loader.dataset)
    depth_mmwave_test_mse = depth_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+mmWave',float(depth_mmwave_test_mse), float(depth_mmwave_test_mpjpe),float(depth_mmwave_test_pampjpe)))
    ### depth + wifi
    depth_wifi_test_mpjpe = depth_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_wifi_test_pampjpe = depth_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_wifi_test_mse = depth_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+WiFi-CSI',float(depth_wifi_test_mse), float(depth_wifi_test_mpjpe),float(depth_wifi_test_pampjpe)))
    ### lidar + mmwave
    lidar_mmwave_test_mpjpe = lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    lidar_mmwave_test_pampjpe = lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    lidar_mmwave_test_mse = lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar+mmWave',float(lidar_mmwave_test_mse), float(lidar_mmwave_test_mpjpe),float(lidar_mmwave_test_pampjpe)))
    ### lidar + wifi
    lidar_wifi_test_mpjpe = lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    lidar_wifi_test_pampjpe = lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    lidar_wifi_test_mse = lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar+WiFi-CSI',float(lidar_wifi_test_mse), float(lidar_wifi_test_mpjpe),float(lidar_wifi_test_pampjpe)))
    ### mmwave + wifi
    mmwave_wifi_test_mpjpe = mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    mmwave_wifi_test_pampjpe = mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    mmwave_wifi_test_mse = mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('mmWave+WiFi-CSI',float(mmwave_wifi_test_mse), float(mmwave_wifi_test_mpjpe),float(mmwave_wifi_test_pampjpe)))
    
    'three modality'
    ### rgb + depth + lidar
    rgb_depth_lidar_test_mpjpe = rgb_depth_lidar_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_test_pampjpe = rgb_depth_lidar_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_test_mse = rgb_depth_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Depth+Lidar',float(rgb_depth_lidar_test_mse), float(rgb_depth_lidar_test_mpjpe),float(rgb_depth_lidar_test_pampjpe)))
    ### rgb + depth + mmwave
    rgb_depth_mmwave_test_mpjpe = rgb_depth_mmwave_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_mmwave_test_pampjpe = rgb_depth_mmwave_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_mmwave_test_mse = rgb_depth_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Depth+mmWave',float(rgb_depth_mmwave_test_mse), float(rgb_depth_mmwave_test_mpjpe),float(rgb_depth_mmwave_test_pampjpe)))
    ### rgb + depth + wifi
    rgb_depth_wifi_test_mpjpe = rgb_depth_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_wifi_test_pampjpe = rgb_depth_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_wifi_test_mse = rgb_depth_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Depth+WiFi-CSI',float(rgb_depth_wifi_test_mse), float(rgb_depth_wifi_test_mpjpe),float(rgb_depth_wifi_test_pampjpe)))
    ### rgb + lidar + mmwave
    rgb_lidar_mmwave_test_mpjpe = rgb_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    rgb_lidar_mmwave_test_pampjpe = rgb_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    rgb_lidar_mmwave_test_mse = rgb_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Lidar+mmWave',float(rgb_lidar_mmwave_test_mse), float(rgb_lidar_mmwave_test_mpjpe),float(rgb_lidar_mmwave_test_pampjpe)))
    ### rgb + lidar + wifi
    rgb_lidar_wifi_test_mpjpe = rgb_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_lidar_wifi_test_pampjpe = rgb_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_lidar_wifi_test_mse = rgb_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Lidar+WiFi-CSI',float(rgb_lidar_wifi_test_mse), float(rgb_lidar_wifi_test_mpjpe),float(rgb_lidar_wifi_test_pampjpe)))
    ### rgb + mmwave + wifi
    rgb_mmwave_wifi_test_mpjpe = rgb_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_mmwave_wifi_test_pampjpe = rgb_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_mmwave_wifi_test_mse = rgb_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+mmWave+WiFi-CSI',float(rgb_mmwave_wifi_test_mse), float(rgb_mmwave_wifi_test_mpjpe),float(rgb_mmwave_wifi_test_pampjpe)))
    ### depth + lidar + mmwave
    depth_lidar_mmwave_test_mpjpe = depth_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_pampjpe = depth_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_mse = depth_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar+mmWave',float(depth_lidar_mmwave_test_mse), float(depth_lidar_mmwave_test_mpjpe),float(depth_lidar_mmwave_test_pampjpe)))
    ### depth + lidar + wifi
    depth_lidar_wifi_test_mpjpe = depth_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_wifi_test_pampjpe = depth_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_wifi_test_mse = depth_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar+WiFi-CSI',float(depth_lidar_wifi_test_mse), float(depth_lidar_wifi_test_mpjpe),float(depth_lidar_wifi_test_pampjpe)))
    ### depth + mmwave + wifi
    depth_mmwave_wifi_test_mpjpe = depth_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_mmwave_wifi_test_pampjpe = depth_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_mmwave_wifi_test_mse = depth_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Depth+mmWave+WiFi-CSI',float(depth_mmwave_wifi_test_mse), float(depth_mmwave_wifi_test_mpjpe),float(depth_mmwave_wifi_test_pampjpe)))
    ### lidar + mmwave + wifi
    lidar_mmwave_wifi_test_mpjpe = lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    lidar_mmwave_wifi_test_pampjpe = lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    lidar_mmwave_wifi_test_mse = lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Lidar+mmWave+WiFi-CSI',float(lidar_mmwave_wifi_test_mse), float(lidar_mmwave_wifi_test_mpjpe),float(lidar_mmwave_wifi_test_pampjpe)))
    
    'four modality'
    ### rgb + depth + lidar + mmwave
    rgb_depth_lidar_mmwave_test_mpjpe = rgb_depth_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_mmwave_test_pampjpe = rgb_depth_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_mmwave_test_mse = rgb_depth_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Depth+Lidar+mmWave',float(rgb_depth_lidar_mmwave_test_mse), float(rgb_depth_lidar_mmwave_test_mpjpe),float(rgb_depth_lidar_mmwave_test_pampjpe)))
    ### rgb + depth + lidar + wifi
    rgb_depth_lidar_wifi_test_mpjpe = rgb_depth_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_wifi_test_pampjpe = rgb_depth_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_wifi_test_mse = rgb_depth_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('RGB+Depth+Lidar+WiFi-CSI',float(rgb_depth_lidar_wifi_test_mse), float(rgb_depth_lidar_wifi_test_mpjpe),float(rgb_depth_lidar_wifi_test_pampjpe)))
    ### rgb + depth + mmwave + wifi
    rgb_depth_mmwave_wifi_test_mpjpe = rgb_depth_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_mmwave_wifi_test_pampjpe = rgb_depth_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_mmwave_wifi_test_mse = rgb_depth_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+Depth+mmWave+WiFi-CSI',float(rgb_depth_mmwave_wifi_test_mse), float(rgb_depth_mmwave_wifi_test_mpjpe),float(rgb_depth_mmwave_wifi_test_pampjpe)))
    ### rgb + lidar + mmwave + wifi
    rgb_lidar_mmwave_wifi_test_mpjpe = rgb_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_lidar_mmwave_wifi_test_pampjpe = rgb_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_lidar_mmwave_wifi_test_mse = rgb_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+Lidar+mmWave+WiFi-CSI',float(rgb_lidar_mmwave_wifi_test_mse), float(rgb_lidar_mmwave_wifi_test_mpjpe),float(rgb_lidar_mmwave_wifi_test_pampjpe)))
    ### depth + lidar + mmwave + wifi
    depth_lidar_mmwave_wifi_test_mpjpe = depth_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_wifi_test_pampjpe = depth_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_wifi_test_mse = depth_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Depth+Lidar+mmWave+WiFi-CSI',float(depth_lidar_mmwave_wifi_test_mse), float(depth_lidar_mmwave_wifi_test_mpjpe),float(depth_lidar_mmwave_wifi_test_pampjpe)))

    'ALL modality'
    ### rgb + depth + lidar + mmwave + wifi
    rgb_depth_lidar_mmwave_wifi_test_mpjpe = rgb_depth_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_mmwave_wifi_test_pampjpe = rgb_depth_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    rgb_depth_lidar_mmwave_wifi_test_mse = rgb_depth_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('RGB+Depth+Lidar+mmWave+WiFi-CSI',float(rgb_depth_lidar_mmwave_wifi_test_mse), float(rgb_depth_lidar_mmwave_wifi_test_mpjpe),float(rgb_depth_lidar_mmwave_wifi_test_pampjpe)))
    return