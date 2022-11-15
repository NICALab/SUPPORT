import torch
import torch.nn as nn

from model.convhole import ConvHole2D

class SUPPORT(nn.Module):
    """
    Blindspot network

    Arguments:
        in_channels: the number of input channels (int)
        mid_channels: the number of middle channels ([int])
    """
    def __init__(self, in_channels, mid_channels=[16, 32, 64, 128, 256], depth=5,\
         blind_conv_channels=64, one_by_one_channels=[32, 16],\
            last_layer_channels=[64, 32, 16], bs_size=3):
        super(SUPPORT, self).__init__()

        # check arguments
        if len(mid_channels) < 2:
            raise Exception("length of mid_channels must be larger than 1")
        if depth % 2 == 0:
            raise Exception("depth must be an odd number")
        if type(blind_conv_channels) != int:
            raise Exception("type of blind_conv_channels must be an integer")
        if not all([type(i)==int for i in one_by_one_channels]):
            raise Exception("one_by_one_channels must be an integer array")

        self.in_channels = in_channels
        self.out_channels = 1
        self.mid_channels = mid_channels
        self.depth = depth
        self.depth3x3 = depth
        self.depth5x5 = depth - 2
        
        self.blind_conv_channels = blind_conv_channels
        self.one_by_one_channels = one_by_one_channels

        self.last_layer_channels = last_layer_channels
        
        self.bs_size = bs_size

        # initialize
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.maxpool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_2d = nn.Upsample(scale_factor=2)

        self._gen_unet()
        self._gen_bsnet()
        
        # last layer
        last_layers = []
        for idx, c in enumerate(last_layer_channels):
            if idx == 0:
                last_layers.append(nn.Conv2d(2*one_by_one_channels[-1], c, kernel_size=1, padding=0))
            else:
                last_layers.append(nn.Conv2d(last_layer_channels[idx-1], c, kernel_size=1, padding=0))
        last_layers.append(nn.Conv2d(c, self.out_channels, kernel_size=1, padding=0))

        self.last_layers = nn.ModuleList(last_layers)

    def _gen_unet(self):
        # (Unet) encoding layers
        self.enc_layers = []
        for i in range(len(self.mid_channels)):
            if i == 0:
                self.enc_layers.append(nn.Conv2d(self.in_channels-1, self.mid_channels[i], kernel_size=3, padding=1))
            else:
                self.enc_layers.append(nn.Conv2d(self.mid_channels[i-1], self.mid_channels[i], kernel_size=3, padding=1))
        self.enc_layers = nn.ModuleList(self.enc_layers)

        # (Unet) decoding layers
        self.dec_layers = []
        for i in range(len(self.mid_channels)-1):
            self.dec_layers.append(nn.Conv2d(self.mid_channels[i] + self.mid_channels[i+1], self.mid_channels[i], kernel_size=3, padding=1))
        self.dec_layers = nn.ModuleList(reversed(self.dec_layers))

        # (Unet) 1x1 convs
        self.unet_1_convs = []
        for idx, c in enumerate(self.one_by_one_channels):
            if idx == 0:
                self.unet_1_convs.append(nn.Conv2d(self.mid_channels[0], c, kernel_size=1, padding=0))
            else:
                self.unet_1_convs.append(nn.Conv2d(self.one_by_one_channels[idx-1], c, kernel_size=1, padding=0))
        self.unet_1_convs = nn.ModuleList(self.unet_1_convs)

    def _gen_bsnet(self):
        # (BS) additional parameters
        self.scalars_3x3 = []
        # c_in_first = self.one_by_one_channels[-1] + 1
        for d in range(self.depth3x3 - 1):
            c_in = 1 if d == 0 else self.blind_conv_channels
            self.scalars_3x3.append(nn.Parameter(torch.ones(c_in), requires_grad=True))
        self.scalars_3x3 = nn.ParameterList(self.scalars_3x3)

        self.scalars_5x5 = []
        for d in range(self.depth5x5 - 1):
            c_in = 1 if d == 0 else self.blind_conv_channels
            self.scalars_5x5.append(nn.Parameter(torch.ones(c_in), requires_grad=True))
        self.scalars_5x5 = nn.ParameterList(self.scalars_5x5)

        # (BS) first layer to process Unet output
        conv3x3 = []
        conv3x3.append(
            nn.Conv2d(
                self.one_by_one_channels[-1],
                self.blind_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                padding_mode="zeros",
                dilation=1,
            )
        )
        conv3x3.append(self.relu)
        self.conv3x3 = nn.ModuleList(conv3x3)

        conv5x5 = []
        conv5x5.append(
            nn.Conv2d(
                self.one_by_one_channels[-1],
                self.blind_conv_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
                padding_mode="zeros",
                dilation=1,
            )
        )
        conv5x5.append(self.relu)
        self.conv5x5 = nn.ModuleList(conv5x5)

        # (BS) dilated convolutions with blind-spot
        blind_conv3x3_layers = []
        for d in range(self.depth3x3):
            c_in = 1 if d == 0 else self.blind_conv_channels
            # """
            pd = pow(2, d)
            if d == self.depth3x3 - 1:
                pd = pd + self.bs_size // 2
            # """
            # pd = pow(2, d)*(self.bs_size//2+1)
            blind_conv3x3_layers.append(
                ConvHole2D(
                    c_in,
                    self.blind_conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=pd, # pow(2, d)*(self.bs_size//2+1),
                    bias=True,
                    padding_mode="zeros",
                    dilation=pd, # pow(2, d)*(self.bs_size//2+1),
                )
            )
            blind_conv3x3_layers.append(self.relu)
        self.blind_convs3x3 = nn.ModuleList(blind_conv3x3_layers)

        blind_conv5x5_layers = []
        for d in range(self.depth5x5):
            c_in = 1 if d == 0 else self.blind_conv_channels
            # """
            pd = pow(3, d)
            if d == self.depth5x5 - 1: # assume we will use only last layer if the bs_size is larger than 1
                pd = pd + self.bs_size // 2
            # """
            # pd = (pow(3, d)*(self.bs_size//2+1))
            blind_conv5x5_layers.append(
                ConvHole2D(
                    c_in,
                    self.blind_conv_channels,
                    kernel_size=5,
                    stride=1,
                    padding=pd * 2, # (pow(3, d)*(self.bs_size//2+1)) * 2,
                    bias=True,
                    padding_mode="zeros",
                    dilation=pd, # *(self.bs_size//2+1),
                )
            )
            blind_conv5x5_layers.append(self.relu)
        self.blind_convs5x5 = nn.ModuleList(blind_conv5x5_layers)

        # (BS) 1x1 convolutions
        out_convs =[]
        for idx, c in enumerate(self.one_by_one_channels):
            if self.bs_size == 1:
                c_in = (
                    self.blind_conv_channels * (self.depth3x3 + self.depth5x5)
                    if idx == 0
                    else self.one_by_one_channels[idx - 1]
                )
            else:
                c_in = (
                    self.blind_conv_channels * 2
                    if idx == 0
                    else self.one_by_one_channels[idx - 1]
                )
            out_convs.append(
                nn.Conv2d(
                    c_in,
                    c,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )
            out_convs.append(self.relu)
        self.out_convs = nn.ModuleList(out_convs)

    def forward_unet(self, x):
        # x = [b, T, d1, d2]
        # d1, d2 = 512, paper reference
        xs = []
        # print(x.size(), x.min(), x.max(), 'unet')

        for idx, enc_layer in enumerate(self.enc_layers):
            x = self.relu(enc_layer(x))
            if idx != len(self.enc_layers) - 1:
                xs.append(x)
                x = self.maxpool_2d(x)

        for idx, dec_layer in enumerate(self.dec_layers):
            # up_ = self.upsample_2d(x)
            # print(xs[-idx-1].size(), xs[-idx-1].size()[2:])
            up_ = torch.nn.functional.interpolate(x, xs[-idx-1].size()[2:])

            x = torch.cat([up_, xs[-idx-1]], dim=1)
            x = self.relu(dec_layer(x))
        
        for one_conv in self.unet_1_convs:
            x = self.relu(one_conv(x))

        return x

    def forward_bsnet(self, x, unet_out):
        # x : bsnet input
        # unet_out : unet output

        hc = []

        unet_out1 = self.conv3x3[0](unet_out)
        unet_out1 = self.conv3x3[1](unet_out1)
        for c in range(self.depth3x3):
            if c == 0:
                x1 = x
            else:
                # x1 = x1 + x1.max() * inp
                # print(x.size())
                x1 = x1 + (self.scalars_3x3[c - 1] * x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            x1 = self.blind_convs3x3[2 * c](x1)
            x1 = self.blind_convs3x3[2 * c + 1](x1)

            if c == 0:
                x1 = x1 + unet_out1
            
            if self.bs_size == 1:
                hc.append(x1)
            else:
                if c == self.depth3x3 - 1:
                    hc.append(x1)

        unet_out2 = self.conv5x5[0](unet_out)
        unet_out2 = self.conv5x5[1](unet_out2)
        for c in range(self.depth5x5):
            if c == 0:
                x2 = x
            else:
                x2 = x2 + (self.scalars_5x5[c - 1] * x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            x2 = self.blind_convs5x5[2 * c](x2)
            x2 = self.blind_convs5x5[2 * c + 1](x2)

            if c == 0:
                x2 = x2 + unet_out2

            if self.bs_size == 1:
                hc.append(x2)
            else:
                if c == self.depth5x5 - 1:
                    hc.append(x2)

        x = torch.cat(hc, dim=1)

        for o_m in self.out_convs:
            x = o_m(x)

        return x

    def forward(self, x):
        # x = [b, T, d1, d2]
        # d1, d2 = 512, paper reference
        
        unet_in = torch.cat([x[:, :self.in_channels//2, :, :], x[:, self.in_channels//2 + 1:, :, :]], dim=1)
        bsnet_in = torch.unsqueeze(x[:, self.in_channels//2, :, :], dim=1)

        unet_out = self.forward_unet(unet_in)
        bsnet_out = self.forward_bsnet(bsnet_in, unet_out)

        x = torch.cat([unet_out, bsnet_out], dim=1)

        for idx, layer in enumerate(self.last_layers):
            if idx != len(self.last_layers)-1:
                x = self.relu(layer(x))
            else:
                x = layer(x)
        
        return x
