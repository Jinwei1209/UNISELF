import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def truncated_normal_(tensor, mean=0, std=1):

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def ConvBlock(
    input_dim, 
    output_dim, 
    kernel_size=3,
    stride=1,
    padding=1, 
    use_bn=1,
    res_connect=0
):

    layers = []

    layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding))
    if use_bn == 1:
        layers.append(nn.BatchNorm2d(output_dim))
    elif use_bn == 2:
        layers.append(nn.GroupNorm(output_dim, output_dim))
    elif use_bn == 4:
        layers.append(AdaINBlock(output_dim))
    elif use_bn == 6:
        layers.append(RasteBlock(output_dim))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding))
    if use_bn == 1:
        layers.append(nn.BatchNorm2d(output_dim))
    elif use_bn == 2:
        layers.append(nn.GroupNorm(output_dim, output_dim))
    elif use_bn == 4:
        layers.append(AdaINBlock(output_dim))
    elif use_bn == 6:
        layers.append(RasteBlock(output_dim))

    if not res_connect:
        layers.append(nn.ReLU(inplace=True))

    return layers


class AdaINBlock(nn.Module):

    def __init__(
        self,
        input_dim,
        num_contrasts = 4
    ):  
        super(AdaINBlock, self).__init__()
        self.input_dim = input_dim
        self.num_contrasts = num_contrasts
        self.contrast_codes = []
        self.generate_binary_codes(self.num_contrasts)
        self.contrast_code_matcher = {code: idx for idx, code in enumerate(self.contrast_codes)}
        self.instanceNorms = nn.ModuleList([nn.GroupNorm(input_dim, input_dim) for _ in self.contrast_codes])

    def generate_binary_codes(self, x, code=""):
        if x == 0:
            self.contrast_codes.append(code)
        else:
            self.generate_binary_codes(x - 1, code + "0")
            self.generate_binary_codes(x - 1, code + "1")

    def forward(self, inputs, input_contrast_code):
        idx = self.contrast_code_matcher.get(input_contrast_code)
        return self.instanceNorms[idx](inputs)


class RasteBlock(nn.Module):
    '''
        Rater Style Encoding (Raste) module
    '''
    def __init__(
        self,
        output_dim
    ):
        super(RasteBlock, self).__init__()
        self.output_dim = output_dim  # number of channels for convolutional layer
        self.group_norm_no_affine = nn.GroupNorm(output_dim, output_dim, affine=False)
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, self.output_dim*2)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.Tanh()

    def forward(self, feature_maps, lambda_value):
        # faeture_maps: feature maps after convolutional layer
        # lambda_value: lambda_value used in rater mixup, lambda_value * rater1 + (1-lambda_value) * rater2
        feature_maps = self.group_norm_no_affine(feature_maps)
        affine_params = self.fc2(self.act(self.fc1(lambda_value)))
        weights = affine_params[:, :self.output_dim, None, None]
        bias = affine_params[:, self.output_dim:, None, None]
        return weights * feature_maps + bias


class DownConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1, 
        pool=True,
        res_connect=0,
    ):

        super(DownConvBlock, self).__init__()
        self.layers = []
        self.use_bn = use_bn
        self.res_connect = res_connect

        if self.res_connect:
            if pool:
                if use_bn == 1:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(input_dim, output_dim, 1, 2, 0),
                        nn.BatchNorm2d(output_dim),
                    )
                elif use_bn == 2:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(input_dim, output_dim, 1, 2, 0),
                        nn.GroupNorm(output_dim, output_dim),
                    )
            else:
                if use_bn == 1:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(input_dim, output_dim, 1, 1, 0),
                        nn.BatchNorm2d(output_dim),
                    )
                elif use_bn == 2:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(input_dim, output_dim, 1, 1, 0),
                        nn.GroupNorm(output_dim, output_dim),
                    )

            self.relu = nn.ReLU(inplace=True)

        if pool:
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            # self.layers.append(nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2, padding=0))

        convBlock = ConvBlock(input_dim, output_dim, kernel_size, stride, padding, use_bn, res_connect)

        for layer in convBlock:
            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)

        self.layers.apply(init_weights)

        if self.res_connect:
            self.shortcut.apply(init_weights)

    def forward(self, inputs, input_contrast_code='1111', lambda_value=None):

        if lambda_value == None:
            lambda_value = torch.tensor([[0.5]]).cuda()

        if self.use_bn == 4:
            x = inputs
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, AdaINBlock):
                    x = layer(x, input_contrast_code)
                else:
                    x = layer(x)
            return x

        elif self.use_bn == 6:
            x = inputs
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, RasteBlock):
                    x = layer(x, lambda_value)
                else:
                    x = layer(x)
            return x

        else:
            if not self.res_connect:
                return self.layers(inputs)

            else:
                identity = inputs
                outputs = self.layers(inputs)
                outputs = outputs + self.shortcut(identity)
                return self.relu(outputs)


class UpConvBlock(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim,
        kernel_size=3,
        stride=1,
        padding=1, 
        use_bn=1,
        use_deconv=0,
        res_connect=0
    ):
        
        super(UpConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_deconv = use_deconv

        if self.use_deconv:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
        else:
            self.upconv_layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, use_bn=use_bn, pool=False, res_connect=res_connect)

    def forward(self, right, left, input_contrast_code='1111', lambda_value=None):
        
        if self.use_deconv:
            right = self.upconv_layer(right)
        else:
            right = nn.functional.interpolate(right, mode='nearest', scale_factor=2)
            right = self.upconv_layer(right)
        
        left_shape = left.size()
        right_shape = right.size()
        padding = (left_shape[3] - right_shape[3], 0, left_shape[2] - right_shape[2], 0)

        right_pad = nn.ConstantPad2d(padding, 0)
        right = right_pad(right)
        out = torch.cat([right, left], 1)
        out =  self.conv_block(out, input_contrast_code, lambda_value)

        return out

