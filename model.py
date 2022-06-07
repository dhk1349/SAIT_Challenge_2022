import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_layer=3, feat_ch=32):
        super(Network, self).__init__()
        self.num_layer = num_layer
        self.feat_ch = feat_ch
        self.conv = nn.Sequential(nn.Conv2d(4, self.feat_ch, 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.feat_ch, int(self.feat_ch / 2), 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(int(self.feat_ch / 2), int(self.feat_ch / 4), 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(int(self.feat_ch / 4), int(self.feat_ch / 8), 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(int(self.feat_ch / 8), int(self.feat_ch / 16), 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(int(self.feat_ch / 16), int(self.feat_ch / 32), 3, 1, 1))

        self.conv.apply(self.weight_init_xavier_uniform)

    def weight_init_xavier_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
            submodule.bias.data.zero_()

    def forward(self, imgs):
        output = self.conv(imgs)

        return output
