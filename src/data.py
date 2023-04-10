import os
import random

import numpy as np
import torch
import torch.utils.data

from utils import calc_rc


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        files,
        config,
        test_loader=False,
    ):
        # フレームシフトなし
        self.window_length = config.window_length
        self.lpc_seq_len = config.seq_len // config.window_length
        self.seq_len = config.window_length * self.lpc_seq_len

        self.hop_length = config.hop_length
        self.test_loader = test_loader

        feature_dir = os.path.join(data_root, "feature")
        self.data = self.read_wavs(feature_dir, files)
        self.len = len(self.data)

        # add

        self.lpc_hop_size = config.hop_size
        self.lpc_frame_size = config.frame_size
        self.lpc_order = config.lpc_order
        self.window = config.window

    def read_wavs(self, feature_dir, files):
        data = []

        for file in files:
            data_dir = os.path.join(feature_dir, file)
            if not os.path.isdir(data_dir):
                continue

            wav = np.load(os.path.join(data_dir, "wav.npy")).astype(np.float32)
            data.append(wav)

        return data

    def pad_noise(self, wav, pad):
        noise = [random.uniform(0.1, 0.3) for i in range(pad)]
        np.place(wav, wav == 0, noise)
        # return wav

    def __getitem__(self, index):
        wav = self.data[index]

        if not self.test_loader:
            amplitude = random.uniform(0.3, 1.0)
            wav = wav * amplitude

            rc_feature, z, gain = calc_rc(wav, self.lpc_order, self.lpc_frame_size,
                                          self.lpc_hop_size)
            pad = self.window_length - np.shape(z)[0] % self.window_length
            z = np.pad(z, (0, pad), 'constant')

            # max_start = np.shape(rc_feature)[0] - self.lpc_seq_len
            # フレームシフトあり
            max_start = np.shape(rc_feature)[0] - self.lpc_seq_len
            if max_start < 0:
                rc_feature = np.pad(rc_feature, ((0, 0), (0, -max_start)), 'constant')

                # add
                gain = np.pad(gain, ((0.0), (0, -max_start)), 'constant')

            else:
                start = random.randint(0, max_start)
                wav_start = start * self.window_length
                # wav_start = self.lpc_hop_size * (start - 1) + self.lpc_frame_size
                rc_feature = rc_feature[start:start + self.lpc_seq_len, :]
                wav = wav[wav_start:wav_start + self.seq_len]

                # add
                gain = gain[start:start + self.lpc_seq_len, :]
                z = z[wav_start:wav_start + self.seq_len]
        else:
            # if np.shape(wav)[0] % self.window_length != 0:
            # pad = self.window_length - np.shape(wav)[0] % self.window_length
            # wav = np.pad(wav, (0, pad), 'constant')
            # self.pad_noise(wav, pad)
            # rc_feature, z = calc_lsp(wav, self.lpc_order, self.lpc_frame_size,
            #                           self.lpc_hop_size)

            # self.pad_noise(wav, pad)
            rc_feature, z, gain = calc_rc(wav, self.lpc_order, self.lpc_frame_size,
                                          self.lpc_hop_size)
            pad = self.window_length - np.shape(z)[0] % self.window_length
            z = np.pad(z, (0, pad), 'constant')
        # else:
        #     rc_feature, z, gain = calc_rc(wav, self.lpc_order, self.lpc_frame_size,
        #                                   self.lpc_hop_size)

        wav = torch.from_numpy(wav).float().unsqueeze(0)
        # rc_feature = torch.from_numpy(rc_feature).float()
        rc_feature = torch.from_numpy(rc_feature).float()
        z = torch.from_numpy(z).float().unsqueeze(0)

        gain = torch.from_numpy(gain).float()

        return wav, rc_feature, z, gain

    def __len__(self):
        return len(self.data)
