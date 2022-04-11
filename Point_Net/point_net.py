import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self):
        super().__init()
        self.identity = nn.Identity

    def forward(self, x):
        self.identity + x


class InputTransform(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config=config
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.config['conv'][0]),(1,3),(1,1),
            nn.BatchNorm2d(self.config['conv'][0]),
            nn.Conv2d(self.config['conv'][0], self.config['conv'][1],(1,1),(1,1)),
            nn.BatchNorm2d(self.config['conv'][1]),
            nn.Conv2d(self.config['conv'][1], self.config['conv'][2],(1,1),(1,1)),
            nn.BatchNorm2d(self.config['conv'][2]),
            nn.MaxPool2d()
            )
        self.linear_layers = nn.Linear

    def forward(self, x):
        B, C, H, W = x.shape # batch_size, num_channels, extra_dim, num_points
        conv = self.conv_layers(x)
        pool = F.max_pool2d(conv, (B, 1))
        pool = pool.reshape(B, -1)
        print("input trfrm bottleneck-", pool.shape)
        bottleneck_dim = pool.shape[-1]
        fc1 = self.linear_layers(bottleneck_dim, self.config['fc'][0])(pool)
        fc2 = self.linear_layers(self.config['fc'][0], self.config['fc'][1])(fc1)
        inp_transform = ResBlock()(fc2, x)
        print("InputTrnsform, Input: {x.shape}, ResInput: {fc2.shape}")
        return inp_transform


class FeatureTransform(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.in_dim=in_dim
        self.config=config
        self.conv_layers = nn.Sequential(
            nn.Conv2d(8, self.config['conv'][3],(1,1), (1,1)),  ### change bsize to a variable
            nn.BatchNorm2d(self.config['conv'][3]),
            nn.Conv2d(self.config['conv'][3], self.config['conv'][4],
                        (1,1), (1,1)),
            nn.BatchNorm2d(self.config['conv'][4]),

        )
        self.linear_layers = nn.Sequential(
                nn.Linear(33, self.config['fc'][2]),  ### bottleneck dim should go here
                nn.Linear(self.config['fc'][2],self.config['fc'][3]),
        )

    def forward(self, x):
        B, C, H, W = x.shape # batch_size, num_channels, extra_dim, num_points
        print("input to feature transform", x.shape)
        assert W > (B+C), "incorrect input shape"
        conv = self.conv_layers(x)
        pool = F.max_pool2d(conv, kernel_size=(B, 1))
        pool = pool.reshape(B, -1)

        bottleneck_dim = pool.shape[-1]
        linear = self.linear_layers(pool)
        print("FeatTrnsform, Input: {x.shape}, ResInput: {fc2.shape}")
        return self.res_block(linear, x)
    
    
class PointNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.config= config
        self.num_classes = num_classes
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(1, self.config['conv'][5],(1,1),(1,1)),
            nn.BatchNorm2d(self.config['conv'][5]),
            nn.Conv2d(self.config['conv'][5], self.config['conv'][6], 
                        (1,1), (1,1)),
            nn.BatchNorm2d(self.config['conv'][6]),
            nn.Conv2d(self.config['conv'][6], self.config['conv'][7],
                        (1,1), (1,1)),
            nn.BatchNorm2d(self.config['conv'][7]))

        self.conv_layers2 = nn.Sequential(
                    nn.Conv2d(self.config['conv'][8], self.config['conv'][9], 
                        (1,1), (1,1)),
                    nn.BatchNorm2d(self.config['conv'][9]),
                    nn.Conv2d(self.config['conv'][9], self.config['conv'][10], 
                        (1,1), (1,1))
        )
        self.linear_layers = nn.Sequential(
                    nn.Linear(self.config['fc'][3], self.config['fc'][4]),
                    nn.BatchNorm2d(self.config['fc'][4]),
                    nn.Linear(self.config['fc'][4], self.config['fc'][4]),
                    nn.Dropout(p=0.7),
                    nn.Linear(self.config['fc'][4], self.num_classes)
        )
        #conv = [64, 128, 128, 512, 2048]
        #fc = [256, 256, 101]

    def forward(self, x):
        # input transform
        B, N, D = x.shape
        print("initial_input shape", x.shape)
        inp_transform = InputTransform(self.config)(x)
        conv1 = self.conv_layers1(inp_transform)

        feat_transform=FeatureTransform(self.config)(conv1)
        print("feat_transform_out", feat_transform.shape)
        B, C, H, W = feat_transform.shape

        conv2 = self.conv_layers2(feat_transform)
        pool = F.max_pool2d(conv2, (N, 1))
        # classification network
        pool_reshaped = pool.reshape(B, -1)
        out = self.linear_layers(pool_reshaped)
        return out


