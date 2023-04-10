import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from entmax import entmax15
from torch.nn.utils import spectral_norm, weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def SNConv1d(*args, **kwargs):
    return spectral_norm(nn.Conv1d(*args, **kwargs))


def SNConvTranspose1d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose1d(*args, **kwargs))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class NeuralCELP(nn.Module):

    def __init__(self, codebook_size, hidden_channels, window_length):
        super().__init__()

        self.codebook_size = codebook_size
        self.hidden_channels = hidden_channels
        self.window_length = window_length

        self.flatten_channels = hidden_channels * window_length // 16
        self.layers = nn.Sequential(
            WNConv1d(1, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2),
            WNConv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            WNConv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            WNConv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            WNConv1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_channels, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.codebook_size),
        )

        self.codebook = nn.Linear(self.codebook_size, self.window_length, bias=False)
        # std = np.sqrt(2.0 / (self.codebook_size + self.window_length))
        # val = np.sqrt(3.0) * std
        val = 0.9
        self.codebook.weight.data.uniform_(-val, val)

    def forward(self, x):
        b, _, t = x.shape
        num_frame = t // self.window_length

        x = self.layers(x)
        x = x.permute(0, 2, 1).reshape(b * num_frame, -1, self.flatten_channels)

        logits = self.proj(x)
        index = entmax15(logits, dim=1, k=self.codebook_size)
        x_hat = self.codebook(index)

        x_hat = x_hat.reshape(b, 1, -1)

        return x_hat


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fms = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fms.append(x)

        x = self.conv_post(x)
        fms.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fms


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = DiscriminatorS(use_spectral_norm=True)

        self.models = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
        ])

        downsamples = []
        sr = config.sampling_rate
        for scale in config.discriminator_scales:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr // scale)
            downsamples.append(resample)
            sr = sr // scale

        self.downsamples = nn.ModuleList(downsamples)

    def forward(self, x):
        logit, fm = self.model(x)
        logits = [logit]
        fms = [fm]

        for disc, resample in zip(self.models, self.downsamples):
            x = resample(x)
            logit, fm = disc(x)
            logits.append(logit)
            fms.append(fm)

        return logits, fms


class DiscriminatorP(nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5,
                                                                                          1), 0))),
            norm_f(
                nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(
                nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5,
                                                                                         1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, x):
        logits = []
        fms = []
        for d in self.discriminators:
            logit, fm = d(x)
            logits.append(logit)
            fms.append(fm)

        return logits, fms
