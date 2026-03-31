from typing import Dict, Tuple

import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, modality_info:Dict, proj_dim):
        '''
        modalitys_info: dict[str, tuple[int, int, int]] tuple of (input_dim, output_dim, feature_num_dim)
        '''
        super().__init__()
        self.projects = nn.ModuleDict()
        for modality, (input_dim, output_dim, feature_num_dim) in modality_info.items():
            #X-Fi's linear projection
            self.projects[modality] = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(feature_num_dim, proj_dim),
            nn.ReLU()
        )
            

    def forward(self, x:Dict):
        features = {}
        for modality, feature in x.items():
            if modality in self.projects:
                features[modality] = self.projects[modality](feature).permute(0, 2, 1) 
            else:
                raise ValueError(f"Modality {modality} not found in projects.")
        return features


class Encoder(nn.Module):
    def __init__(self, index:int, encoder: nn.Module, info: Tuple):
        super().__init__()
        self.index = index
        self.encoder = encoder
        self.info = info

    def forward(self, x):
        return self.encoder(x)

class Encoders(nn.Module):
    def __init__(self, encoders:nn.ModuleDict, proj_dim=32):
        super().__init__()       
        self.encoders = encoders
        modalities_info = {key: encoder.info for key, encoder in encoders.items()}
        self.proj = LinearProjection(modalities_info, proj_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        features = {}
        for key, input in x.items():

            features[key] = self.encoders[key](input)
        projected_features = self.proj(features)
        return projected_features

    def get_index(self, modality:str):
        return self.encoders[modality].index

class Decoder(nn.Module):
    def __init__(self, decoder: nn.Module, task_type=None):
        super().__init__()
        self.decoder = decoder
        self.task_type = task_type

    def forward(self, x):
        return self.decoder(x)

