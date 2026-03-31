import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime
import time

def generate_none_empth_modality_list():
    modality_list = random.choices(
        [True, False],
        k=1,
        weights=[50, 50]
    )
    wifi_ = random.choices(
        [True, False],
        k=1,
        weights=[90, 10]
    )
    modality_list.append(wifi_[0])
    rfid_ = random.choices(
        [True, False],
        k=1,
        weights=[60, 40]
    )
    modality_list.append(rfid_[0])
    if sum(modality_list) == 0:
        modality_list = generate_none_empth_modality_list()
        return modality_list
    else:
        return modality_list


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    labels = []
    [labels.append(float(t[3])) for t in batch]
    labels = torch.FloatTensor(labels)

    # mmwave
    mmwave_data = np.array([(t[2]) for t in batch])
    mmwave_data = torch.FloatTensor(mmwave_data)

    # wifi-csi
    wifi_data = np.array([(t[0]) for t in batch])
    wifi_data = torch.FloatTensor(wifi_data)

    # rfid
    rfid_data = np.array([(t[1]) for t in batch])
    rfid_data = torch.FloatTensor(rfid_data)

    modality_list = generate_none_empth_modality_list()

    return mmwave_data, wifi_data, rfid_data, labels, modality_list


def har_test(model, tensor_loader, criterion, device, val_random_seed):
    model.eval()
    test_acc = 0
    test_loss = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        mmwave_data, wifi_data, rfid_data, labels, modality_list = data
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        rfid_data = rfid_data.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        outputs = model(mmwave_data, wifi_data, rfid_data, modality_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * labels.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("Validation Accuracy:{:.4f}, Loss:{:.5f}".format(float(test_acc), float(test_loss)))
    return test_acc


def har_train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, save_dir,
              val_random_seed):
    optimizer = torch.optim.AdamW(
        [
            {'params': model.linear_projector.parameters()},
            {'params': model.X_Fusion_block.parameters()}
        ],
        lr=learning_rate
    )
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        random.seed(epoch)
        for i, data in enumerate(tqdm(train_loader)):
            mmwave_data, wifi_data, rfid_data, labels, modality_list = data
            mmwave_data = mmwave_data.to(device)
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = model(mmwave_data, wifi_data, rfid_data, modality_list)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)
            loss = criterion(outputs, labels)
            if loss == float('nan'):
                print('nan')
                print(outputs)
                print(labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = epoch_accuracy / len(train_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
        if (epoch + 1) % 5 == 0:
            test_acc = har_test(
                model=model,
                tensor_loader=test_loader,
                criterion=criterion,
                device=device,
                val_random_seed=val_random_seed
            )
            if test_acc >= best_test_acc:
                print(f"Best test accuracy is:{test_acc}")
                best_test_acc = test_acc
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion, device):
    model.eval()

    test_loss = {'mmwave': 0, 'wifi': 0, 'rfid': 0,
                 'mmwave+wifi': 0, 'mmwave+rfid': 0, 'wifi+rfid': 0,
                 'all': 0}
    total_correct = {k: 0 for k in test_loss.keys()}
    total_samples = 0

    time_data_loading = 0.0
    time_to_device = 0.0
    time_model_inference = 0.0

    with torch.no_grad():
        loop_start_time = time.time()

        for data in tqdm(tensor_loader, desc="Final Testing"):
            data_load_end_time = time.time()
            time_data_loading += data_load_end_time - loop_start_time

            mmwave_data, wifi_data, rfid_data, labels = data

            transfer_start_time = time.time()
            mmwave_data = mmwave_data.to(device)
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels = labels.long().to(device)
            torch.cuda.synchronize()
            transfer_end_time = time.time()
            time_to_device += transfer_end_time - transfer_start_time

            inputs = {'mmwave': mmwave_data, 'wifi': wifi_data, 'rfid': rfid_data}
            total_samples += labels.size(0)

            for mode_str in test_loss.keys():
                inference_start_time = time.time()
                outputs, _, _ = model(inputs, val=True, mode=mode_str)
                torch.cuda.synchronize()
                inference_end_time = time.time()
                time_model_inference += inference_end_time - inference_start_time

                loss = criterion(outputs, labels)
                test_loss[mode_str] += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct[mode_str] += (predicted == labels).sum().item()

            loop_start_time = time.time()

    print("\n" + "=" * 30)
    print("      Final Multi-modal Evaluation Results (Legacy)")
    print("=" * 30)
    for mode in test_loss.keys():
        avg_loss = test_loss[mode] / total_samples
        accuracy = total_correct[mode] / total_samples
        print(f"Modality: {mode:<15} | Average Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
    print("=" * 30)

    print("\n" + "=" * 30)
    print("        Performance Analysis (Time Consumption)")
    print("=" * 30)
    print(f"Total Data Loading Time:   {time_data_loading:.4f} seconds")
    print(f"Total Data Transfer Time (CPU -> GPU):    {time_to_device:.4f} seconds")
    print(f"Total Model Inference Time:     {time_model_inference:.4f} seconds")
    print("=" * 30)
