import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from Encoders import Encoders, Decoder
from misc import random_num_select
from HAR.losses.diffkd.diffkd import DiffKD
from HAR.losses.dist_kd import *


class KD_Weights(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.kd_weights = nn.Parameter(torch.rand(shape), requires_grad=True)  # kd weights as teachers
        # self.kd_weights = nn.Parameter(torch.ones(shape), requires_grad=True)

    def get_kd_weights(self):
        # return self.kd_weights
        # return torch.sigmoid(self.kd_weights)
        return F.softmax(self.kd_weights, dim=0)

    def forward(self, i, j):
        # kd_w = self.kd_weights[i] / self.kd_weights[j]
        # kd_w = torch.sigmoid(self.kd_weights[i]) / torch.sigmoid(self.kd_weights[j])
        kd_w = self.get_kd_weights()[i] / self.get_kd_weights()[j]
        return kd_w


class DualNet(nn.Module):

    def __init__(self, task_encoders: nn.ModuleDict, task_decoder: Tuple,
                 self_att=False, cross_att=False, proj_dim=32):
        """
        @param task_encoders: task encoders: nn.ModuleDict[str, Encoder]
        @param task_decoder: task decoder: Tuple[Decoder, task_type]
        @param self_att:
        @param cross_att:
        @param proj_dim:
        """
        super().__init__()
        self.encoders = Encoders(task_encoders, proj_dim)
        self.shared_dec = Decoder(task_decoder[0], task_decoder[1])
        self.modalities = list(task_encoders.keys())
        self.modality_num = len(self.modalities)

        # MHA
        self.self_att = self_att
        self.cross_att = cross_att
        embed_dim = 512
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8).cuda()
        self.layer_norm = nn.LayerNorm(embed_dim).cuda()

        self.kd_weights = KD_Weights(shape=self.modality_num)

        latent_dim = embed_dim // 2
        original_diff_kd_module = DiffKD(
            student_channels=embed_dim,
            teacher_channels=embed_dim,
            use_ae=True,
            ae_channels=latent_dim
        )
        self.diff_kd_module = nn.ModuleDict({
            mod_name: copy.deepcopy(original_diff_kd_module)
            for mod_name in self.modalities
        })
        # self.distillation_loss = nn.L1Loss()
        self.distillation_loss = DIST(beta=1.0, gamma=1.0, tau=1.0)

    # best
    def forward(self, inputs: Dict, val=False, mode="all"):
        """
        @param inputs: multi-modality input: Dict[modality_name, tensor_data]
        @param val: train or val, if val calculate not calculate kdloss
        @param mode: all or random, (val or train) if random
        @return: task_output, select_modality, kd_loss
        """
        # random choose modes instead of specifying
        select_input_key, other_input_key = [], []

        if mode == "random":
            select_input_key, other_input_key, _ = random_num_select(list(inputs.keys()))
        elif mode == "all":
            select_input_key = list(inputs.keys())
            other_input_key = []
        else:
            select_input_key = [m for m in self.modalities if m in mode.split('+')]
            other_input_key = [m for m in self.modalities if m not in select_input_key]

        # encode selected modalities
        assert select_input_key, f"Mode '{mode}' is invalid or led to no selected inputs."

        select_inputs = {
            modality: inputs[modality] 
            for modality in select_input_key 
            if modality in self.encoders.encoders
        }
        if len(select_inputs) != len(select_input_key):
             raise ValueError("One or more selected modalities not found in encoders.")

        shared_features_dict = self.encoders(select_inputs)

        teacher_feature = None
        if not val and len(select_input_key) > 1:
            weights = self.kd_weights.get_kd_weights()
            weighted_features = [
                shared_features_dict[mod] * weights[self.encoders.get_index(mod)]
                for mod in select_input_key
            ]
            teacher_feature = torch.stack(weighted_features).sum(dim=0)

        features_dict = {k: v.clone() for k, v in shared_features_dict.items()}
        tot_kd_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if not val and teacher_feature is not None:
            refined_students_for_fusion = []
            teacher_for_diffkd = teacher_feature.permute(0, 2, 1)

            for mod in select_input_key:
                student_feature = shared_features_dict[mod]
                student_for_diffkd = student_feature.permute(0, 2, 1)

                specific_module = self.diff_kd_module[mod]

                refined_student_latent, target_teacher_latent, diff_loss, ae_loss = \
                    specific_module(student_feat=student_for_diffkd, teacher_feat=teacher_for_diffkd.detach())

                distill_loss = self.distillation_loss(refined_student_latent, target_teacher_latent.detach())

                tot_kd_loss = tot_kd_loss + distill_loss + diff_loss

                if ae_loss is not None:
                    tot_kd_loss = tot_kd_loss + ae_loss

                decoded_feature_bdl = specific_module.ae.decoder(refined_student_latent)
                refined_students_for_fusion.append(decoded_feature_bdl.permute(0, 2, 1))

            for i, mod_name in enumerate(select_input_key):
                features_dict[mod_name] = refined_students_for_fusion[i]

            if select_input_key:
                tot_kd_loss = tot_kd_loss / len(select_input_key)

        if other_input_key and select_input_key:
            all_weights = self.kd_weights.get_kd_weights()
            
            weighted_features = []
            weights_for_avg = []
            
            for mod_name in select_input_key:
                mod_idx = self.encoders.get_index(mod_name)
                mod_weight = all_weights[mod_idx]
                
                weighted_features.append(features_dict[mod_name] * mod_weight)
                weights_for_avg.append(mod_weight)
                
            sum_of_weighted_features = torch.stack(weighted_features).sum(dim=0)
            sum_of_weights = torch.stack(weights_for_avg).sum()
            avg_feature = sum_of_weighted_features / (sum_of_weights + 1e-8)
            
            for mod_name in other_input_key:
                features_dict[mod_name] = avg_feature

        tensor_list = [features_dict[m] for m in self.modalities]
        features_tensor = torch.stack(tensor_list, dim=1)  # (N, M, Fn, D)

        if val:
            weights = self.kd_weights.get_kd_weights().view(1, features_tensor.shape[1], 1, 1)
            features_tensor = features_tensor * weights

        if self.self_att:
            # self attention
            N, M, Fn, D = features_tensor.shape
            attn_input = features_tensor.view(N, M * Fn, D)
            out_features, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        else:
            N, M, Fn, D = features_tensor.shape
            out_features = features_tensor.view(N, M * Fn, D)
        
        pred = self.shared_dec(out_features)

        return pred, select_input_key, tot_kd_loss.mean()

