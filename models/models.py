import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _pair
import numpy as np
import os

## quantizatin conv layers

_NBITS = 8
_ACTMAX = 4.0

class LinQuantSteOp(torch.autograd.Function):
    """
    Straight-through estimator operator for linear quantization
    """

    @staticmethod
    def forward(ctx, input, signed, nbits, max_val):
        """
        In the forward pass we apply the quantizer
        """
        assert max_val > 0
        if signed:
            int_max = 2 ** (nbits - 1) - 1
        else:
            int_max = 2 ** nbits
        scale = max_val / int_max
        return input.div(scale).round_().mul_(scale)


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output, None, None, None

quantize = LinQuantSteOp.apply


class Conv2dQuant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.nbits = _NBITS
        self.input_signed = False
        self.input_quant = True
        self.input_max = _ACTMAX

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.input_quant:
            max_val = self.input_max
            if self.input_signed:
                min_val = -max_val
            else:
                min_val = 0.0
            input = quantize(input.clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)

        max_val = self.weight.abs().max().item()
        weight = quantize(self.weight, True, self.nbits, max_val)
        return self.conv2d_forward(input, weight)


class ConvTrans2dQuant(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(ConvTrans2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias,
                 dilation, padding_mode)
        self.nbits = _NBITS
        self.input_signed = False
        self.input_quant = True
        self.input_max = _ACTMAX

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        if self.input_quant:
            max_val = self.input_max
            if self.input_signed:
                min_val = -max_val
            else:
                min_val = 0.0
            input = quantize(input.clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)

        max_val = self.weight.abs().max().item()
        weight = quantize(self.weight, True, self.nbits, max_val)
        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



class ResidualBlock(nn.Module):
    def __init__(self, in_features=256, mid_features=256, conv_class=nn.Conv2d):
        super(ResidualBlock, self).__init__()
        
        if mid_features > 0:
            conv_block = [  nn.ReflectionPad2d(1),
                        conv_class(in_features, mid_features, 3),
                        nn.InstanceNorm2d(mid_features, affine=True),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        conv_class(mid_features, in_features, 3),
                        nn.InstanceNorm2d(in_features, affine=False)  ]

            self.conv_block = nn.Sequential(*conv_block)
        else:
            self.conv_block = nn.Sequential()

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, dim_lst=None, alpha=1, quant=False):
        '''
        Args:
            dim_lst: channel dimensions. [int]
            alpha: channel width factor. float. 
        '''

        super(Generator, self).__init__()
        if quant:
            print('!!! Quantized model !!!')

        if quant:
            conv_class = Conv2dQuant
            transconv_class = ConvTrans2dQuant
        else:
            conv_class = nn.Conv2d
            transconv_class = nn.ConvTranspose2d

        if dim_lst is None:
            dim_lst = [64, 128] + [256]*n_residual_blocks + [128, 64]
        if alpha is not 1:
            dim_lst = (np.array(dim_lst) * alpha).astype(int).tolist()
        print(dim_lst)
        assert len(dim_lst) == 4 + n_residual_blocks

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    conv_class(input_nc, dim_lst[0], 7),
                    nn.InstanceNorm2d(dim_lst[0], affine=True),
                    nn.ReLU(inplace=True) ]

        # do not quantize the input image
        if conv_class is Conv2dQuant:
            model[1].input_quant = False

        # Downsampling
        model += [  conv_class(dim_lst[0], dim_lst[1], 3, stride=2, padding=1),
                    nn.InstanceNorm2d(dim_lst[1], affine=True),
                    nn.ReLU(inplace=True) ]
        
        model += [  conv_class(dim_lst[1], int(256*alpha), 3, stride=2, padding=1),
                    nn.InstanceNorm2d(int(256*alpha), affine=False),
                    nn.ReLU(inplace=True) ]

        # Residual blocks
        for i in range(n_residual_blocks):
            model += [ResidualBlock(in_features=int(256*alpha), mid_features=dim_lst[2+i], conv_class=conv_class)]

        # Upsampling
        model += [  transconv_class(int(256*alpha), dim_lst[2+n_residual_blocks], 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(dim_lst[2+n_residual_blocks], affine=True),
                    nn.ReLU(inplace=True) ]
        model += [  transconv_class(dim_lst[2+n_residual_blocks], dim_lst[2+n_residual_blocks+1], 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(dim_lst[2+n_residual_blocks+1], affine=True),
                    nn.ReLU(inplace=True) ]

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    conv_class(dim_lst[2+n_residual_blocks+1], output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.print_networks()

    def forward(self, x):
        return self.model(x)
    
    def print_networks(self):
        print('---------- Networks initialized -------------')
        # for name in self.model_names:
            # if hasattr(self, name):
                # net = getattr(self, name)
        name = 'netG'
        net = self.model
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels),
            norm_layer(in_channels * scale_factor,affine=True),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.conv(x)

class MobileResnetBlock(nn.Module):
    def __init__(self, ic, oc, padding_type, norm_layer, dropout_rate):
        super(MobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(ic, oc, padding_type, norm_layer, dropout_rate)

    def build_conv_block(self, ic, oc, padding_type, norm_layer, dropout_rate):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=ic, out_channels=oc,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(oc, affine=True),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=oc, out_channels=ic,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(ic, affine=True)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SubMobileResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, dim_lst=None, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_residual_blocks=9, padding_type='reflect', alpha=1):
        assert n_residual_blocks >= 0
        super(SubMobileResnetGenerator, self).__init__()

        if dim_lst is None:
            dim_lst = [16, 32, 128] + [128, 64, 64, 64, 128, 128, 128 ,128 ,128] + [128, 32, 16]
        if alpha is not 1:
            dim_lst = (np.array(dim_lst) * alpha).astype(int).tolist()
        print(dim_lst)
        assert len(dim_lst) == 6 + n_residual_blocks
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, dim_lst[0], kernel_size=7, padding=0),
                 norm_layer(dim_lst[0], affine=True),
                 nn.ReLU(True)]

        model += [  nn.Conv2d(dim_lst[0], dim_lst[1], 3, stride=2, padding=1),
                    nn.InstanceNorm2d(dim_lst[1], affine=True),
                    nn.ReLU(inplace=True) ]
        
        model += [  nn.Conv2d(dim_lst[1], dim_lst[2], 3, stride=2, padding=1),
                    nn.InstanceNorm2d(dim_lst[2], affine=True),
                    nn.ReLU(inplace=True) ]

        for i in range(n_residual_blocks):
            model += [MobileResnetBlock(dim_lst[2], dim_lst[i + 4], padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate )]


        # mult = 2 ** n_downsampling

        # ic = config['channels'][2]
        # oc = config['channels'][3]
        # for i in range(n_residual_blocks):
        #     oc = config['channels'][i + 3]
        #     model += [MobileResnetBlock(ic * mult, oc * mult, padding_type=padding_type, norm_layer=norm_layer,
        #                                 dropout_rate=dropout_raten_residual_blocks)]
        # for i in range(n_residual_blocks):
        #     if len(config['channels']) == 6:
        #         offset = 0
        #     else:
        #         offset = i // 3
        #     oc = config['channels'][offset + 3]
        #     model += [MobileResnetBlock(ic * mult, oc * mult, padding_type=padding_type, norm_layer=norm_layer,
        #                                 dropout_rate=dropout_rate )]
        # Upsampling
        model += [  nn.ConvTranspose2d(int(dim_lst[3+n_residual_blocks]), dim_lst[4+n_residual_blocks], 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(dim_lst[4+n_residual_blocks], affine=True),
                    nn.ReLU(inplace=True) ]
        model += [  nn.ConvTranspose2d(dim_lst[4+n_residual_blocks], dim_lst[5+n_residual_blocks], 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(dim_lst[5+n_residual_blocks], affine=True),
                    nn.ReLU(inplace=True) ]

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(dim_lst[5+n_residual_blocks], output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.print_networks()
        # if len(config['channels']) == 6:
        #     offset = 4
        # else:
        #     offset = 6
        # for i in range(n_downsampling):  # add upsampling layers
        #     oc = config['channels'][offset + i]
        #     # print(oc)
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ic * mult, int(oc * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(oc * mult / 2)),
        #               nn.ReLU(True)]
        #     ic = oc
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ic, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]
        # self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
    def print_networks(self):
        print('---------- Networks initialized -------------')
        # for name in self.model_names:
            # if hasattr(self, name):
                # net = getattr(self, name)
        name = 'netG'
        net = self.model
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))

class Discriminator(nn.Module):
    def __init__(self, input_nc, in_affine=True):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, affine=in_affine), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256, affine=in_affine), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512, affine=in_affine), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == "__main__":
    # from utils.utils import measure_model, model_size
    # dim_lst = [16, 32, 128] + [128, 64, 64, 64, 128, 128, 128 ,128 ,128] + [128, 32, 16]
    dim_lst = [32, 64, 128] + [128, 128, 128, 128, 128, 128, 128 ,128 ,128] + [128, 64, 32]
    g = SubMobileResnetGenerator(3, 3, dim_lst=dim_lst)
    # measure_model(g, 256, 256)
    # print(model_size(g))
