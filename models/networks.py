"""
Encoder, decoder, transformation, router, and dense layer architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def actvn(x):
    return F.leaky_relu(x, negative_slope=0.3)

class EncoderSmall(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EncoderSmall, self).__init__()

        self.dense1 = nn.Linear(in_features=input_shape, out_features=4*output_shape, bias=False)
        self.bn1 = nn.BatchNorm1d(4*output_shape)
        self.dense2 = nn.Linear(in_features=4*output_shape, out_features=4*output_shape, bias=False)
        self.bn2 = nn.BatchNorm1d(4*output_shape)
        self.dense3 = nn.Linear(in_features=4*output_shape, out_features=2*output_shape, bias=False)
        self.bn3 = nn.BatchNorm1d(2*output_shape)
        self.dense4 = nn.Linear(in_features=2*output_shape, out_features=output_shape, bias=False)
        self.bn4 = nn.BatchNorm1d(output_shape)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = actvn(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = actvn(x)
        return x, None, None

class DecoderSmall(nn.Module):
    def __init__(self, input_shape, output_shape, activation):
        super(DecoderSmall, self).__init__()
        self.activation = activation
        self.dense1 = nn.Linear(in_features=input_shape, out_features=128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(in_features=128, out_features=256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dense3 = nn.Linear(in_features=256, out_features=512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dense4 = nn.Linear(in_features=512, out_features=512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.dense5 = nn.Linear(in_features=512, out_features=output_shape, bias=True)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = actvn(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = actvn(x)
        x = self.dense5(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class EncoderSmallCnn(nn.Module):
    def __init__(self, encoded_size):
        super(EncoderSmallCnn, self).__init__()
        n_maps_output = encoded_size//4
        self.cnn0 = nn.Conv2d(in_channels=1, out_channels=n_maps_output//4, kernel_size=3, stride=2, padding=0, bias=False)
        self.cnn1 = nn.Conv2d(in_channels=n_maps_output//4, out_channels=n_maps_output//2, kernel_size=3, stride=2, padding=0, bias=False)
        self.cnn2 = nn.Conv2d(in_channels=n_maps_output//2, out_channels=n_maps_output, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(n_maps_output//4)
        self.bn1 = nn.BatchNorm2d(n_maps_output//2)
        self.bn2 = nn.BatchNorm2d(n_maps_output)

    def forward(self, x):
        x = self.cnn0(x)
        x = self.bn0(x)
        x = actvn(x)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = actvn(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = x.view(x.size(0), -1)
        return x, None, None

class DecoderSmallCnn(nn.Module):
    def __init__(self, input_shape, activation):
        super(DecoderSmallCnn, self).__init__()
        self.activation = activation
        self.dense = nn.Linear(in_features=input_shape, out_features=3 * 3 * 32, bias=False)
        self.bn = nn.BatchNorm1d(3 * 3 * 32)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.cnn1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, bias=False)
        self.cnn2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)        
        self.cnn3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = actvn(x)
        x = x.view(-1, 32, 3, 3)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = actvn(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.cnn3(x)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class EncoderOmniglot(nn.Module):
    def __init__(self, encoded_size):
        super(EncoderOmniglot, self).__init__()
        self.cnns = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=encoded_size//4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//4, out_channels=encoded_size//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//4, out_channels=encoded_size//2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//2, out_channels=encoded_size//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//2, out_channels=encoded_size, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size, out_channels=encoded_size, kernel_size=5, bias=False)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(encoded_size//4),
            nn.BatchNorm2d(encoded_size//4),
            nn.BatchNorm2d(encoded_size//2),
            nn.BatchNorm2d(encoded_size//2),
            nn.BatchNorm2d(encoded_size),
            nn.BatchNorm2d(encoded_size)
        ])

    def forward(self, x):
        for i in range(len(self.cnns)):
            x = self.cnns[i](x)
            x = self.bns[i](x)
            x = actvn(x)
        x = x.view(x.size(0), -1)
        return x, None, None
        
class DecoderOmniglot(nn.Module):
    def __init__(self, input_shape, activation):
        super(DecoderOmniglot, self).__init__()
        self.activation = activation
        self.dense = nn.Linear(in_features=input_shape, out_features=2 * 2 * 128, bias=False)
        self.cnns = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, bias=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=0, output_padding=1, bias=False)
        ])
        self.cnns.append(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True))
        self.bn = nn.BatchNorm1d(2 * 2 * 128)
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32)
        ])

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = actvn(x)
        x = x.view(-1, 128, 2, 2)
        for i in range(len(self.bns)):
            x = self.cnns[i](x)
            x = self.bns[i](x)
            x = actvn(x)
        x = self.cnns[-1](x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()

        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(in_channels=fin, out_channels=self.fhidden, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=self.fhidden, out_channels=self.fout, kernel_size=3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels=fin, out_channels=self.fout, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.fin)
        self.bn1 = nn.BatchNorm2d(self.fhidden)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.bn0(x)))
        dx = self.conv_1(actvn(self.bn1(dx)))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class Resnet_Encoder(nn.Module):
    def __init__(self, s0=2, nf=8, nf_max=256, size=32):
        super(Resnet_Encoder, self).__init__()

        self.s0 = s0 
        self.nf = nf  
        self.nf_max = nf_max
        self.size = size

        # Submodules
        nlayers = int(torch.log2(torch.tensor(size / s0).float()))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, kernel_size=3, padding=1)

        self.resnet = nn.Sequential(*blocks)

        self.bn0 = nn.BatchNorm2d(self.nf0)


    def forward(self, x):
        out = self.conv_img(x)
        out = self.resnet(out)
        out = actvn(self.bn0(out))
        out = out.view(out.size(0), -1)
        return out, None, None
    
class Resnet_Decoder(nn.Module):
    def __init__(self, s0=2, nf=8, nf_max=256, ndim=64, activation='sigmoid', size=32):
        super(Resnet_Decoder, self).__init__()

        self.s0 = s0
        self.nf = nf  
        self.nf_max = nf_max 
        self.activation = activation

        # Submodules
        nlayers = int(torch.log2(torch.tensor(size / s0).float()))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        self.fc = nn.Linear(ndim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ]
        blocks += [
            ResnetBlock(nf, nf),
        ]
        self.resnet = nn.Sequential(*blocks)

        self.bn0 = nn.BatchNorm2d(nf)
        self.conv_img = nn.ConvTranspose2d(nf, 3, kernel_size=3, padding=1)


    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(self.bn0(out)))
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out


# Small branch transformation
class MLP(nn.Module):
    def __init__(self, input_size, encoded_size, hidden_unit):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_unit, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_unit)
        self.mu = nn.Linear(hidden_unit, encoded_size)
        self.sigma = nn.Linear(hidden_unit, encoded_size)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return x, mu, sigma


class Dense(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(Dense, self).__init__()
        self.mu = nn.Linear(input_size, encoded_size)
        self.sigma = nn.Linear(input_size, encoded_size)

    def forward(self, inputs):
        x = inputs
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Router(nn.Module):
    def __init__(self, input_size, hidden_units=128):
        super(Router, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_units, bias=False)
        self.dense2 = nn.Linear(hidden_units, hidden_units, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.dense3 = nn.Linear(hidden_units, 1)

    def forward(self, inputs, return_last_layer=False):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        d = F.sigmoid(self.dense3(x))
        if return_last_layer:
            return d, x
        else:
            return d
        

def get_encoder(architecture, encoded_size, x_shape, size=None):
    if architecture == 'mlp':
        encoder = EncoderSmall(input_shape=x_shape, output_shape=encoded_size)
    elif architecture == 'cnn1':
        encoder = EncoderSmallCnn(encoded_size)
    elif architecture == 'cnn2':
        encoder = Resnet_Encoder(s0=4, nf=32, nf_max=256, size=size)
    elif architecture == 'cnn_omni':
        encoder = EncoderOmniglot(encoded_size)
    else:
        raise ValueError('The encoder architecture is mispecified.')
    return encoder


def get_decoder(architecture, input_shape, output_shape, activation):
    if architecture == 'mlp':
        decoder = DecoderSmall(input_shape, output_shape, activation)
    elif architecture == 'cnn1':
        decoder = DecoderSmallCnn(input_shape, activation)
    elif architecture == 'cnn2':
        size = int((output_shape/3)**0.5)
        decoder = Resnet_Decoder(s0=4, nf=32, nf_max=256, ndim = input_shape, activation=activation, size=size)
    elif architecture == 'cnn_omni':
        decoder = DecoderOmniglot(input_shape, activation) 
    else:
        raise ValueError('The decoder architecture is mispecified.')
    return decoder
