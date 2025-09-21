import torch
import torch.nn as nn
import torchvision.models as models

class HybridATCNet(nn.Module):
    def __init__(self, num_numeric_features=13, pretrained=True, output_dim=1, dropout_prob=0.3):
        super(HybridATCNet, self).__init__()
        # CNN for side images
        self.side_cnn = models.resnet18(pretrained=pretrained)
        self.side_cnn.fc = nn.Identity()
        self.side_embed_dim = 512

        # CNN for back images
        self.back_cnn = models.resnet18(pretrained=pretrained)
        self.back_cnn.fc = nn.Identity()
        self.back_embed_dim = 512

        # Fully connected layers
        self.fc1 = nn.Linear(self.side_embed_dim + self.back_embed_dim + num_numeric_features, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, side_img, back_img, numeric_features):
        side_feat = self.side_cnn(side_img)
        back_feat = self.back_cnn(back_img)
        x = torch.cat([side_feat, back_feat, numeric_features], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
