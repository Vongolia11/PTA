import os.path as osp
import timeit

from tensorboardX import SummaryWriter

from DualNet import DualNet, F
from Encoders import *
from engine import *
from Extractor import mmwave_feature_extractor, wifi_feature_extractor, rfid_feature_extractor, classification_Head
from misc import *
from HAR.backbone_models.mmWave.ResNet import *

class Task:
    def __init__(self, task_name: str, modalities: List[str]):
        super().__init__()
        self.task_name = task_name
        self.modalities = modalities
        self.encoders = nn.ModuleDict()
        self.decoder = None
        self.losses = dict()


    def train(self, parser):
        with Engine(custom_parser=parser) as engine:
            args = parser.parse_args()
            if args.num_gpus > 1:
                torch.cuda.set_device(args.local_rank)

            writer = SummaryWriter(args.snapshot_dir)
            seed = args.random_seed

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            self.set_encoders()
            self.set_decoder()
            self.set_losses()

            model = DualNet(self.encoders, (self.decoder, "classification"), self_att=False, cross_att=False)
            model.train()
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)
            # optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if 'kd_weights' not in n],
            #                               args.learning_rate, momentum=0.99, nesterov=True)
            optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if 'kd_weights' not in n],
                                         lr=args.learning_rate, weight_decay=args.weight_decay)
            kd_optim = torch.optim.Adam(model.kd_weights.parameters(), args.learning_rate,
                                        weight_decay=args.weight_decay)

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

            if args.reload_from_checkpoint:
                print('Loading weights: {}'.format(args.reload_path))
                if os.path.exists(args.reload_path):
                    # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
                    checkpoint = torch.load(args.reload_path)
                    model = checkpoint['model']
                    optimizer = checkpoint['optimizer']
                    kd_optim = checkpoint['kd_optim']
                    args.start_iters = checkpoint['iter']
                    print("Loaded model trained for", args.start_iters, "iters")
                else:
                    print('File not exists in the reload path: {}'.format(args.reload_path))
                    exit(0)

            train_loader, val_loader = self.set_train_data()
            train_iter = iter(train_loader)
            for i_iter in range(args.start_iters, args.num_steps):
                try:
                    data = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    data = next(train_iter)

                mmwave_data, wifi_data, rfid_data, labels = data

                mmwave_data = mmwave_data.to(device)
                wifi_data = wifi_data.to(device)
                rfid_data = rfid_data.to(device)
                labels = labels.to(device)
                labels = labels.long()

                inputs = {
                    'mmwave': mmwave_data,
                    'wifi': wifi_data,
                    'rfid': rfid_data}

                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)
                outputs, _, tot_kd_loss = model(inputs, mode="random")
                outputs = outputs.float()

                task_loss = self.losses["ce"](outputs, labels)
                total_loss = task_loss + 0.1 * tot_kd_loss
                total_loss.backward()
                optimizer.step()
                print(f"Recognition Loss: {task_loss.item()}, KD Loss: {tot_kd_loss.item()}")

                # write log per 100 iter
                if i_iter % 100 == 0 and (args.local_rank == 0):
                    writer.add_scalar('learning_rate', lr, i_iter)
                    writer.add_scalar('loss', total_loss.cpu().data.numpy(), i_iter)

                print('iter = {} of {} completed, lr = {:.4}, task_loss = {:.4}, kd_loss = {:.4}'
                      .format(i_iter, args.num_steps, lr, task_loss.cpu().data.numpy(),
                              tot_kd_loss.cpu().data.numpy()))

                # save model in time
                if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
                    print('save last model ...')
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer,
                        'kd_optim': kd_optim,
                        'iter': i_iter
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'xrf55_final.pth'))
                    break

                # save model when val
                if i_iter % args.val_pred_every == args.val_pred_every - 1 and i_iter != 0 and (args.local_rank == 0):
                    print('save model ...')
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer,
                        'kd_optim': kd_optim,
                        'iter': i_iter
                    }
                    # torch.save(checkpoint, osp.join(args.snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'xrf55_last.pth'))

                if not args.train_only and i_iter % args.val_pred_every == 0:
                    val_start = timeit.default_timer()
                    
                    val_loss, val_acc = self.validate(args, device, model, val_loader, kd_optim)

                    if args.local_rank == 0:
                        writer.add_scalar('Val_Loss', val_loss, i_iter)
                        writer.add_scalar('Val_Accuracy', val_acc, i_iter)

                        print(
                            f'--- Validation complete, Iter = {i_iter}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2%} ---')
                        print(
                            f"Current kd_weights (after Softmax): {F.softmax(model.kd_weights.kd_weights, dim=0).detach().cpu().numpy()}")

                    val_end = timeit.default_timer()
                    print(f'Validation took {val_end - val_start:.2f} seconds')


    def validate(self, args, device, model, val_loader, kd_optim):
        model.eval()

        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0

        for index, data in enumerate(val_loader):
            mmwave_data, wifi_data, rfid_data, labels = data

            mmwave_data = mmwave_data.to(device)
            wifi_data = wifi_data.to(device)
            rfid_data = rfid_data.to(device)
            labels = labels.long().to(device)

            inputs = {
                'mmwave': mmwave_data,
                'wifi': wifi_data,
                'rfid': rfid_data
            }

            kd_optim.zero_grad()
            pred, _, _ = model(inputs, val=True, mode="all")
            pred = pred.float()
            loss = self.losses["ce"](pred, labels)
            loss.backward()
            kd_optim.step()

            total_val_loss += loss.item() * labels.size(0)

            _, predicted_labels = torch.max(pred, 1)
            total_samples += labels.size(0)
            total_correct += (predicted_labels == labels).sum().item()
            print(
                f"Current kd_weights (after Softmax): {F.softmax(model.kd_weights.kd_weights, dim=0).detach().cpu().numpy()}")

        avg_loss = total_val_loss / total_samples
        accuracy = total_correct / total_samples

        print('Printing Validation Results:')
        print(f'Average Validation Loss: {avg_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2%}')

        model.train()

        return avg_loss, accuracy

    def set_encoders(self):
        mmwave_model = torch.load('./HAR/backbone_models/mmWave/mmwave_ResNet18.pt')
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)
        mmwave_extractor.eval()
        wifi_model = torch.load('./HAR/backbone_models/WIFI/wifi_ResNet18.pt')
        wifi_extractor = wifi_feature_extractor(wifi_model)
        wifi_extractor.eval()
        rfid_model = torch.load('./HAR/backbone_models/RFID/rfid_ResNet18.pt')
        rfid_extractor = rfid_feature_extractor(rfid_model)
        rfid_extractor.eval()

        encoders = nn.ModuleList([
            mmwave_extractor,
            wifi_extractor,
            rfid_extractor
        ])

        encode_info = [
            (512, 512, 32),
            (512, 512, 4),
            (512, 512, 5)
        ]

        for i, name in enumerate(self.modalities):
            self.encoders[name] = Encoder(i, encoders[i], encode_info[i])

    def set_decoder(self):
        self.decoder = classification_Head(emb_size=512, num_classes=55)

    def set_losses(self):
        self.losses["ce"] = nn.CrossEntropyLoss()


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

    return mmwave_data, wifi_data, rfid_data, labels
