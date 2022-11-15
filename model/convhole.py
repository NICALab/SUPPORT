import math
import torch
import torch.nn as nn
from torch.nn import init


class Input(nn.Module):
    def __init__(self):
        super(Input, self).__init__()

    def forward(self, x):
        return x


class ConvHole2D(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        kernel_initializer=None,
        device=None,
        dtype=None,
    ):
        super(ConvHole2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert self.groups == 1, "(groups > 1) is Not implemented!"

        # import pdb; pdb.set_trace()

        _w = torch.empty(
            out_channels,
            in_channels // groups,
            self.kernel_size[0] * self.kernel_size[1] - 1,
        )
        if kernel_initializer is None:
            init.kaiming_uniform_(_w, a=math.sqrt(5))
        else:
            nn.init.ones_(_w)

        self._oc, self._ic, self._p = _w.size()
        self._spot = nn.Parameter(torch.zeros([self._oc, self._ic, 1]), requires_grad=False)
        self.weight = nn.Parameter(_w, requires_grad=True)
    
    # overwrite default behavior
    def forward(self, input):
        _kernel = torch.cat(
            [self.weight[:, :, : self._p // 2],
            self._spot,
            self.weight[:, :, self._p // 2 :]],
            dim=2
        )
        _kernel = _kernel.view([self._oc, self._ic, *self.kernel_size])
        return self._conv_forward(input, _kernel, self.bias)



class ConvHole3D(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode="zeros",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        kernel_initializer=None,
        device=None,
        dtype=None,
    ):
        super(ConvHole3D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert self.groups == 1, "(groups > 1) is Not implemented!"

        _w = torch.empty(
            out_channels,
            in_channels // groups,
            self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] - 1,
        )
        if kernel_initializer is None:
            init.kaiming_uniform_(_w, a=math.sqrt(5))
        else:
            nn.init.ones_(_w)


        self._oc, self._ic, self._p = _w.size()
        self._spot = nn.Parameter(torch.zeros([self._oc, self._ic, 1]), requires_grad=False)

        self.weight = nn.Parameter(_w, requires_grad=True)

    # overwrite default behavior
    def forward(self, input):
        _kernel = torch.cat(
            [self.weight[:, :, : self._p // 2],
            self._spot,
            self.weight[:, :, self._p // 2 :]],
            dim=2
        )
        _kernel = _kernel.view([self._oc, self._ic, *self.kernel_size])
        return self._conv_forward(input, _kernel, self.bias)