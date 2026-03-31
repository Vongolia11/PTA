from Encoders import *


class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        # x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 32

class wifi_feature_extractor(nn.Module):
    def __init__(self, wifi_model):
        super(wifi_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(wifi_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        # x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 4

class rfid_feature_extractor(nn.Module):
    def __init__(self, rfid_model):
        super(rfid_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rfid_model.children())[:-3])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        # x = x.permute(0, 2, 1)
        return x 
    # shape: B, 512, 5


class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27):
        super(classification_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.norm(x)
        x = self.fc(x)
        # x = x.view(x.size(0), 17, 3)
        return x
