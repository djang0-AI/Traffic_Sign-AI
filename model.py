import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_classes, conv_scaling = 1, conv_output_size = 1024, fc_width = 128):
    
        super(Net, self).__init__()
    
        self.model = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=int(16*conv_scaling), kernel_size=3, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(16*conv_scaling)),
            nn.Conv2d(in_channels= int(16*conv_scaling), out_channels=int(32*conv_scaling), kernel_size=3, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(32*conv_scaling)),
            nn.MaxPool2d(kernel_size = 2),

            nn.BatchNorm2d(int(32*conv_scaling)),
            nn.Conv2d(in_channels= int(32*conv_scaling), out_channels=int(64*conv_scaling), kernel_size=3, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(64*conv_scaling)),
            nn.Conv2d(in_channels= int(64*conv_scaling), out_channels=int(128*conv_scaling), kernel_size=3, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(128*conv_scaling)),
            nn.MaxPool2d(kernel_size = 2),
                  
            nn.BatchNorm2d(int(128*conv_scaling)),
            nn.Conv2d(in_channels= int(128*conv_scaling), out_channels=int(256*conv_scaling), kernel_size=3, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(256*conv_scaling)),
            nn.Conv2d(in_channels= int(256*conv_scaling), out_channels=conv_output_size, kernel_size=1, stride=1,bias = False),
            nn.Hardswish(),
            nn.BatchNorm2d(int(conv_output_size)),
            nn.AdaptiveMaxPool2d(output_size = 1),
    
            nn.Flatten(),
            nn.BatchNorm1d(conv_output_size),
            nn.Dropout(0.5),
            nn.Linear(in_features=conv_output_size, out_features=fc_width, bias = False),
            nn.BatchNorm1d(fc_width),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(in_features=fc_width, out_features=n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        y = self.model(x)

        return y
