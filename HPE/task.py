import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import sys
import os
from typing import Dict, List
from backbones.RGB_benchmark.RGB_ResNet import RGB_ResNet18
from backbones.depth_benchmark.depth_ResNet18 import Depth_ResNet18
from backbones.mmwave_benchmark.mmwave_point_transformer_TD import mmwave_PointTransformerReg
from backbones.lidar_benchmark.lidar_point_transformer import lidar_PointTransformerReg
from backbones.lidar_benchmark.pointnet_util import farthest_point_sample, index_points

from meta_diffusion.losses.diffkd.diffkd import DiffKD
from meta_diffusion.losses.dist_kd import DIST


class MetaDistillation(nn.Module):
    def __init__(self, modalities: List, embed_dim: int, feature_len: int):
        super().__init__()
        self.modalities = modalities
        self.embed_dim = embed_dim
        self.feature_len = feature_len

        self.diff_kd_module = DiffKD(
            student_channels=self.embed_dim,
            teacher_channels=self.embed_dim,
            use_ae=True,
            ae_channels=self.embed_dim // 2,
            inference_steps=5,
        )
        self.distillation_loss = DIST(beta=1.0, gamma=1.0, tau=1.0)

        self.kd_weights = nn.Parameter(torch.ones(len(self.modalities)), requires_grad=True)

    def get_meta_weights(self):
        return F.softmax(self.kd_weights, dim=-1)

    def forward(self, shared_features_dict: Dict, val: bool = False):
        device = next(iter(shared_features_dict.values())).device
        if not val:
            if len(shared_features_dict) > 1:
                tot_kd_loss = torch.tensor(0.0, device=device)
                weights = self.get_meta_weights()
                active_indices = torch.tensor([self.modalities.index(key) for key in shared_features_dict.keys()], device=device)
                active_weights = weights[active_indices]
                normalized_active_weights = active_weights / active_weights.sum()

                weighted_features = [feat * normalized_active_weights[i] for i, (mod, feat) in enumerate(shared_features_dict.items())]
                teacher_feature = torch.stack(weighted_features).sum(dim=0).permute(0, 2, 1)

                refined_students_decoded = {}
                for mod, student_feature in shared_features_dict.items():
                    student_feature_bdl = student_feature.permute(0, 2, 1)
                    refined_student_latent, target_teacher_latent, diff_loss, ae_loss = \
                        self.diff_kd_module(student_feat=student_feature_bdl, teacher_feat=teacher_feature.detach())

                    B, D_latent, L = refined_student_latent.shape
                    student_for_dist = refined_student_latent.permute(0, 2, 1).reshape(B * L, D_latent)
                    teacher_for_dist = target_teacher_latent.permute(0, 2, 1).reshape(B * L, D_latent)
                    distill_loss = self.distillation_loss(student_for_dist, teacher_for_dist.detach())

                    tot_kd_loss += distill_loss + diff_loss
                    if ae_loss is not None:
                        tot_kd_loss += ae_loss

                    decoded_feature = self.diff_kd_module.ae.decoder(refined_student_latent)
                    refined_students_decoded[mod] = decoded_feature.permute(0, 2, 1)

                out_features = torch.cat(list(refined_students_decoded.values()), dim=1)
                return out_features, tot_kd_loss
            else: 
                out_features = list(shared_features_dict.values())[0]
                return out_features, torch.tensor(0.0, device=device)
        
        else:
            if not shared_features_dict:
                return torch.tensor([], device=device), torch.tensor(0.0, device=device)

            weights = self.get_meta_weights()
            active_indices = torch.tensor([self.modalities.index(key) for key in shared_features_dict.keys()], device=device)
            active_weights = weights[active_indices]
            normalized_active_weights = active_weights / active_weights.sum()

            weighted_features = [feat * normalized_active_weights[i] for i, (mod, feat) in enumerate(shared_features_dict.items())]
            avg_feature = torch.stack(weighted_features).sum(dim=0)
            
            return avg_feature, torch.tensor(0.0, device=device)

class rgb_feature_extractor(nn.Module):
    def __init__(self, rgb_model):
        super(rgb_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rgb_model.children())[:-2])
    def forward(self, x):
        return self.part(x).view(x.size(0), 512, -1).permute(0, 2, 1)

class depth_feature_extractor(nn.Module):
    def __init__(self, depth_model):
        super(depth_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(depth_model.children())[:-2])
    def forward(self, x):
        return self.part(x).view(x.size(0), 512, -1).permute(0, 2, 1)

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-1])
    def forward(self, x):
        x, _ = self.part(x)
        return x

class lidar_feature_extractor(nn.Module):
    def __init__(self, lidar_model):
        super(lidar_feature_extractor, self).__init__()
        self.fc1 = lidar_model.backbone.fc1
        self.transformer1 = lidar_model.backbone.transformer1
        self.transition_downs = lidar_model.backbone.transition_downs
        self.transformers = lidar_model.backbone.transformers
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(len(self.transition_downs)):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
        return points.view(points.size(0), -1, 512)

class csi_feature_extractor(nn.Module):
    def __init__(self, model):
        super(csi_feature_extractor, self).__init__()
        self.part = nn.Sequential(
            model.encoder_conv1, model.encoder_bn1, model.encoder_relu,
            model.encoder_layer1, model.encoder_layer2,
            model.encoder_layer3, model.encoder_layer4,
        )
        self.resize = Resize([136, 32])
    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3).flatten(3, 4)
        x = self.resize(x)
        return self.part(x).view(x.size(0), 512, -1).permute(0, 2, 1)

def selective_pos_enc(xyz, npoint):
    fps_idx = farthest_point_sample(xyz, npoint)
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    return new_xyz

class feature_extrator(nn.Module):
    def __init__(self):
        super(feature_extrator, self).__init__()
        hpe_root = os.path.dirname(os.path.abspath(__file__))
        
        rgb_path = os.path.join(hpe_root, 'backbones/RGB_benchmark/RGB_Resnet18.pt')
        rgb_model = RGB_ResNet18()
        rgb_model.load_state_dict(torch.load(rgb_path, weights_only=True))
        self.rgb_extractor = rgb_feature_extractor(rgb_model).eval()

        depth_path = os.path.join(hpe_root, 'backbones/depth_benchmark/depth_Resnet18.pt')
        depth_model = Depth_ResNet18()
        depth_model.load_state_dict(torch.load(depth_path, weights_only=True))
        self.depth_extractor = depth_feature_extractor(depth_model).eval()

        mmwave_path = os.path.join(hpe_root, 'backbones/mmwave_benchmark/mmwave_all_random_TD.pt')
        mmwave_model = mmwave_PointTransformerReg()
        mmwave_model.load_state_dict(torch.load(mmwave_path, weights_only=True))
        self.mmwave_extractor = mmwave_feature_extractor(mmwave_model).eval()

        lidar_path = os.path.join(hpe_root, 'backbones/lidar_benchmark/lidar_all_random.pt')
        lidar_model = lidar_PointTransformerReg(root=hpe_root)
        lidar_model.load_state_dict(torch.load(lidar_path, weights_only=True))
        self.lidar_extractor = lidar_feature_extractor(lidar_model).eval()

        csi_path = os.path.join(hpe_root, 'backbones/CSI_benchmark/protocol3_random_1.pkl')
        csi_benchmark_dir = os.path.dirname(csi_path)
        if csi_benchmark_dir not in sys.path:
            sys.path.insert(0, csi_benchmark_dir)
        csi_model = torch.load(csi_path, weights_only=False)
        self.csi_extractor = csi_feature_extractor(csi_model).eval()

    def forward(self, rgb_data, depth_data, mmwave_data, lidar_data, csi_data, modality_list):
        real_feature_list = []
        if modality_list[0]: real_feature_list.append(self.rgb_extractor(rgb_data))
        if modality_list[1]: real_feature_list.append(self.depth_extractor(depth_data))
        if modality_list[2]: real_feature_list.append(self.mmwave_extractor(mmwave_data))
        if modality_list[3]: real_feature_list.append(self.lidar_extractor(lidar_data))
        if modality_list[4]: real_feature_list.append(self.csi_extractor(csi_data))
        return real_feature_list

class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim, feature_len=32):
        super(linear_projector, self).__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv1d(input_dim, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(49, feature_len), nn.ReLU()), # RGB
            nn.Sequential(nn.Conv1d(input_dim, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(49, feature_len), nn.ReLU()), # Depth
            nn.Sequential(nn.Conv1d(input_dim, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(32, feature_len), nn.ReLU()), # mmWave
            nn.Sequential(nn.Conv1d(input_dim, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(32, feature_len), nn.ReLU()), # LiDAR
            nn.Sequential(nn.Conv1d(input_dim, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Linear(17*4, feature_len), nn.ReLU()) # CSI
        ])
        self.pos_enc_layer = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, feature_list, lidar_points, modality_list):
        active_indices = [i for i, active in enumerate(modality_list) if active]
        projected_feature_list = [self.projections[i](
            feature_list[j].permute(0, 2, 1)) for j, i in enumerate(active_indices)]

        if not projected_feature_list:
            return torch.tensor([], device=lidar_points.device if lidar_points is not None else 'cpu')
        
        projected_feature_list_permuted = [p.permute(0, 2, 1) for p in projected_feature_list]
        projected_feature_cat = torch.cat(projected_feature_list_permuted, dim=1)

        if modality_list[3] and lidar_points is not None:
            new_xyz = selective_pos_enc(lidar_points, projected_feature_cat.shape[1])
            pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
            projected_feature_cat += pos_enc
        return projected_feature_cat

class regression_Head(nn.Module):
    def __init__(self, emb_size=512, num_classes=17*3):
        super(regression_Head, self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        return x.view(x.size(0), 17, 3)
class Midas(nn.Module):
    def __init__(self, num_modalities=5, feature_dim=512, feature_len=32):
        super(Midas, self).__init__()
        self.feature_extractor = feature_extrator()
        self.linear_projector = linear_projector(
            input_dim=feature_dim, output_dim=feature_dim, feature_len=feature_len)

        self.modalities = ['rgb', 'depth', 'mmwave', 'lidar', 'csi']
        self.fusion_block = MetaDistillation(
            modalities=self.modalities,
            embed_dim=feature_dim,
            feature_len=feature_len
        )
        self.regression_head = regression_Head(
            emb_size=feature_dim,
            num_classes=17*3
        )

    def forward(self, datas: dict, modality_list: list, val: bool = False):
        feature_list = self.feature_extractor(
            datas.get('rgb'), datas.get('depth'), datas.get('mmwave'),
            datas.get('lidar'), datas.get('csi'), modality_list
        )

        if not feature_list:
            batch_size = next((d.shape[0] for d in datas.values() if d is not None), 1)
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 17, 3, device=device), torch.tensor(0.0, device=device)
        
        projected_features = self.linear_projector(
            feature_list, datas.get('lidar'), modality_list)

        if projected_features.nelement() == 0:
            batch_size = next((d.shape[0] for d in datas.values() if d is not None), 1)
            device = next(self.parameters()).device
            return torch.zeros(batch_size, 17, 3, device=device), torch.tensor(0.0, device=device)
        
        active_modalities = [self.modalities[i] for i, active in enumerate(modality_list) if active]

        if projected_features.shape[1] % len(active_modalities) != 0:
            raise ValueError(
                f"Projected feature length ({projected_features.shape[1]}) "
                f"is not divisible by the number of active modalities ({len(active_modalities)})."
            )

        feature_chunks = projected_features.chunk(len(active_modalities), dim=1)

        shared_features_dict = {
            mod: feature_chunks[i] for i, mod in enumerate(active_modalities)
        }
        
        fused_features, kd_loss = self.fusion_block(shared_features_dict, val=val)
        prediction = self.regression_head(fused_features)

        return prediction, kd_loss

