import torch.nn.functional as F
import torch.nn as nn

from .unet2d_blocks import *


class Unet(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels=1,
        num_filters=[2**i for i in range(5, 10)],
        bilateral=0,
        use_deconv=0,
        use_deconv2=0,
        use_bn=2,
        res_connect=0
    ):

        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.bilateral = bilateral
        self.use_bn = use_bn
        
        if self.bilateral:
            self.upsampling_path2 = nn.ModuleList()
            self.output_channels = 1
        
        for i in range(len(self.num_filters)):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.downsampling_path.append(DownConvBlock(input_dim, output_dim, pool=pool, use_bn=use_bn, res_connect=res_connect))

        for i in range(len(self.num_filters)-2, -1, -1):

            input_dim = self.num_filters[i+1]
            output_dim = self.num_filters[i]

            self.upsampling_path.append(UpConvBlock(input_dim, output_dim, use_deconv=use_deconv, use_bn=use_bn, res_connect=res_connect))

            if self.bilateral:
                self.upsampling_path2.append(UpConvBlock(input_dim, output_dim, use_deconv=use_deconv2, use_bn=use_bn, res_connect=res_connect))
        
        self.last_layer = nn.Conv2d(output_dim, self.output_channels, kernel_size=1)
        if self.bilateral:
            self.last_layer2 = nn.Conv2d(output_dim, self.output_channels, kernel_size=1)


    def forward(self, x, input_contrast_code='1111', lambda_value=None):

        blocks = []

        for idx, down in enumerate(self.downsampling_path):
            x = down(x, input_contrast_code, lambda_value)

            if idx != len(self.downsampling_path)-1:
                blocks.append(x)

        for idx, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-idx-1], input_contrast_code, lambda_value)

        del blocks
        
        x = self.last_layer(x)

        return x