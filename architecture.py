import torch.nn as nn
import torch.nn.functional as F

# constants
# size of grid
S = 7
# number of bounding boxes per grid cell
B = 2
# no classes
C = 20


class darknet(nn.Module):
    def __init__(self, batch_norm=True, pretrain=False):
        super(darknet, self).__init__()
        
        self.convolutional_layers = self._make_convolutional_layers(batch_norm)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        # for loop with print size for debugging
        for layer in self.convolutional_layers:
            x = layer(x)
            # print(x.size())
            
#         x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(x.size(0), S, S,  B * 5 + C)
        return x
    
    def _make_convolutional_layers(self, batch_norm):
        if batch_norm:
            convolutional_layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
                nn.Conv2d(64, 192, 3, padding=1),
                nn.BatchNorm2d(192),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 192, 56, 56]
                
                nn.Conv2d(192, 128, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 256, 28, 28]
                
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 512, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 1024, 14, 14]
                
                nn.Conv2d(1024, 512, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 512, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                
    #             [1, 1024, 7, 7]
                
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            convolutional_layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
                nn.Conv2d(64, 192, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 192, 56, 56]
                
                nn.Conv2d(192, 128, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 256, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 256, 28, 28]
                
                nn.Conv2d(512, 256, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 256, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 512, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 512, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                
    #             [1, 1024, 14, 14]
                
                nn.Conv2d(1024, 512, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 512, 1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(512, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                
    #             [1, 1024, 7, 7]
                
                nn.Conv2d(1024, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                
                nn.LeakyReLU(0.1, inplace=True),
            )
        return convolutional_layers