import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile
# import torch
from pysptk import lpc2lsp, lsp2lpc


def levinson():
    pass


def add_noise(wav):
    for i in range(lpc_feature.shape[0]):

        lpc_feature[i, :] = np.place(lpc_feature[i, :], lpc_feature[i, :] == 0, noise)

    return lpc_feature


def lpc_analysis(x, lpc_order, frame_size, hop_size):
    """calculate lpc features from signal.

    Args:
        x (ndarray): signal
        lpc_order (int): lpc order
        frame_size (int): frame size
        hop_size (int): hop size

    Returns:
        lpc_feature (ndarray): LP feature
        resid (ndarray): residual signal
        gain (ndarray): gain
        parcor (ndarray): parcor feature
    """

    twin = scipy.signal.windows.hamming(frame_size, sym=False)  # time window

    # x = np.pad(x, [frame_size // 2, frame_size // 2], 'constant')  # zero padding
    # x = scipy.signal.lfilter([1, 0.97], [1], x) preemphasize speech (not used anymore)
    pad = hop_size - (len(x) - frame_size) % hop_size
    x = np.pad(x, (0, pad), 'constant')

    # nframe = 1 + int((len(x) - frame_size + 1) / hop_size)
    nframe = (len(x) - frame_size) // hop_size + 1

    lpc_feature = np.zeros((nframe, lpc_order + 1))
    err = np.zeros(lpc_order + 1)
    # err = np.zeros(lpc_order)

    gain = np.zeros(nframe)
    parcor = np.zeros((nframe, lpc_order))
    resid = np.zeros((nframe, frame_size))

    fi = 0  # frame index
    for i in range(0, len(x) - frame_size + 1, hop_size):
        xi = twin * x[i:i + frame_size]
        sp = np.abs(np.fft.fft(xi))**2
        r = np.fft.ifft(sp)
        # wr = lwin * r.real  # lag windowing
        wr = r.real

        # lpc analysis (Levinson's method)
        lpc_feature[fi, 0] = 1
        lpc_feature[fi, 1] = -wr[1] / wr[0]
        err[1] = wr[0] + wr[1] * lpc_feature[fi, 1]
        parcor[fi, 0] = lpc_feature[fi, 1]
        # parcor[fi, 0] = 1

        for p in range(1, lpc_order):
            parcor[fi, p] = -np.sum(lpc_feature[fi, 0:p + 1] * wr[p + 1:0:-1]) / err[p]
            U = lpc_feature[fi, 0:p + 2]
            lpc_feature[fi, 0:p + 2] = U + parcor[fi, p] * U[::-1]
            err[p + 1] = err[p] * (1 - parcor[fi, p]**2)

        gain[fi] = np.sqrt(err[lpc_order])  # gain

        # calculate residual
        resid[fi, :] = scipy.signal.lfilter(lpc_feature[fi, :], gain[fi], xi)
        fi += 1

    return lpc_feature, resid, gain, parcor


def lpc_to_lsp(lpc_feature):
    """convert LP feature to LSP feature.

    Args:
        lpc_feature (ndaray): LP feature

    Returns:
        lsp_feature (ndarray): LSP feature
    """
    nframe = lpc_feature.shape[0]
    lpc_order = lpc_feature.shape[1] - 1
    lsp_feature = np.zeros((nframe, lpc_order + 1))

    for i in range(nframe):
        if (np.all(lpc_feature[i, :] == 0)):
            break
        lsp_feature[i, :] = lpc2lsp(lpc_feature[i, :])

    return lsp_feature


def lsp_to_lpc(lsp):
    """convert LSP feature to LP feature. (atten)

    Returns:
        _type_: _description_
    """
    nframe = lsp.shape[0]
    P = lsp.shape[1] - 1
    lpc = np.zeros((nframe, P + 1))

    for i in range(nframe):
        lpc[i, :] = lsp2lpc(lsp[i, :])

    lpc[:, 0] = 1

    return lpc


def lpc2lsp(lpc):
    P = len(lpc)
    a = np.zeros(P + 1)
    a[0:-1] = lpc
    p = np.zeros(P + 1)
    q = np.zeros(P + 1)
    for i in range(P + 1):
        j = P - i
        p[i] = a[i] + a[j]
        q[i] = a[i] - a[j]

    ps = np.sort(np.angle(np.roots(p)))
    qs = np.sort(np.angle(np.roots(q)))
    lsp = np.concatenate((ps[len(ps) // 2:], qs[len(qs) // 2:]), axis=None)
    lsp = np.sort(lsp)[:-1]
    lsp[0] = 0
    return lsp


# def lsp2lpc(lsp):
#     ps = torch.cat(lsp[:,])
# def parcor_to_lpc(parcor):


def fir_filter(b, a, x):
    """FIR filter

    Args:
        b (Tensor): the number of coefficient vector
        a (Tensor): the denominator coefficient vector
        x (Tensor): input array

    Returns:
        xi (Tensor): signal
    """
    frame_size = x.shape[1]
    filter_order = a.shape[1] - 1

    for i in range(frame_size):
        for j in range(filter_order):
            if (i - j) >= 0:
                xi[i] += b[fi] / (a[j] * x[i - j])
    return xi


def lpc_synthesis(lpc_feature, resid, gain, hop_size):
    """synthesize from lpc feature and residual signal on gpu.

    Args:
        lpc_feature (Tensor): lpc
        resid (Tensor): residual signal
        gain (Tensor): gain
        hop_size (int): hop size
    """
    nframe = resid.shape[0]
    frame_size = resid.shape[1]
    lpc_order = lpc_feature.shape[1] - 1

    # z = overlappadd(resid, frame_size, hop_size)
    # win = torch.hamming_window(frame_size)
    xlen = (nframe - 1) * hop_size + frame_size
    framed_x = torch.cat([torch.zeros([nframe, lpc_order]), torch.zeros_like(resid)], axis=1)

    # lpc = lpc_feature[:, ::-1]
    lpc = torch.flip(lpc_feature, dims=[1])
    for idx in range(frame_size):
        framed_x[:, idx + lpc_order] = resid[:, idx] * gain[:]

        pred = torch.sum(framed_x[:, idx:idx + lpc_order] * lpc[:, :-1], axis=1)
        framed_x[:, idx + lpc_order] = framed_x[:, idx + lpc_order] - pred

    x = overlappadd(framed_x[:, lpc_order:], frame_size, hop_size, win)

    return x

    # wav_idx = 0
    # for fi in range(nframe):
    #     xi = torch.zeros(frame_size)

    #     for i in range(frame_size):
    #         xi[i] = 0
    #         for j in range(lpc_order):
    #             if resid[fi, i - j] == 0:
    #                 continue
    #             if (i - j) >= 0:
    #                 xi[i] += gain[fi] / lpc_feature[fi, j] * resid[fi, i - j]
    #     # xi = fir_filter(gain[fi], lpc_feature[fi, :], resid[fi, :])
    #     x[wav_idx:wav_idx + frame_size] += xi[:frame_size + 1] * win  # weighted overlap add
    #     wav_idx += hop_size

    # x *= hop_size / torch.sum(win**2)


# def lpc_synthesis(lpc_feature, resid, gain, hop_size):
#     """synthesize from lpc feature and residual signal.

#     Args:
#         lpc_feature (ndarray): lpc feature
#         resid (ndarray): residual signal
#         gain (ndarray): gain
#         hop_size (int): hop size

#     Returns:
#         x (ndarray): signal.
#     """
#     nframe = resid.shape[0]
#     frame_size = resid.shape[1]

#     win = np.hamming(frame_size)  # window
#     xlen = (nframe - 1) * hop_size + frame_size
#     x = np.zeros(xlen)

#     i = 0
#     for fi in range(nframe):
#         xi = scipy.signal.lfilter([gain[fi]], lpc_feature[fi, :],
#                                   resid[fi, :])  # synthesize speech from Llpc_order coeffs

#         x[i:i + frame_size] += xi[:frame_size + 1] * win  # weighted overlap add
#         i += hop_size

#     x = x * hop_size / np.sum(win**2)  # scaled weighted overlap add

#     # x = x[frame_size // 2:]

#     return x


def framing(x, frame_size, hop_size):
    """framing singal

    Args:
        x (ndarray): signal
        frame_size (int): frame size
        hop_size (int): hop size

    Returns:
        framed_x (ndarray): framed signal
    """

    nframe = (x.shape[0] - frame_size) // hop_size + 1
    framed_x = np.zeros([nframe, frame_size], dtype=x.dtype)
    for frame_idx in range(nframe):
        framed_x[frame_idx, :] = x[frame_idx * hop_size:frame_idx * hop_size + frame_size]

    return framed_x


def overlappadd(framed_x, frame_size, hop_size, window):
    xlen = (framed_x.shape[0] - 1) * hop_size + frame_size
    x = np.zeros(xlen)
    # x = torch.zeros(xlen)

    # window = tor.hamming(frame_size)

    for frame_idx in range(framed_x.shape[0]):
        wav_idx = frame_idx * hop_size
        x[wav_idx:wav_idx + frame_size] += framed_x[frame_idx, :] * window

    x = x * hop_size / torch.sum(window**2)
    # x = x[frame_size // 2:]

    return x


def main():

    import json
    import sys

    import librosa
    import matplotlib.pyplot as plt
    import scipy.io.wavfile
    from attrdict import AttrDict

    with open("config.json", 'r') as f:
        config = AttrDict(json.load(f))

    wav_path = "data/wav/jvs001/BASIC5000_0025.wav"
    save_path = "tmp/jvs001_BASIC5000_0025.wav"
    wav, _ = librosa.core.load(wav_path, sr=config.sampling_rate)
    wav = librosa.util.normalize(wav) * 0.99
    wav, _ = librosa.effects.trim(wav, top_db=100, frame_length=2048, hop_length=512)
    wav = wav.astype(np.float32)

    lpc, f_resid, gain, _ = lpc_analysis(wav, config.lpc_order, config.frame_size, config.hop_size)
    win = torch.hamming_window(config.frame_size)

    # f_resid = torch.from_numpy(f_resid.astype(np.float32)).clone()

    z = overlappadd(f_resid, config.frame_size, config.hop_size, win)

    # _t = lpc_synthesis(lpc, f_resid, gain, config.hop_size)

    lsp = lpc_to_lsp(lpc)

    _lpc = lsp_to_lpc(lsp)
    _f_resid = framing(z, config.frame_size, config.hop_size)

    # numpy->torch
    # _lpc = torch.from_numpy(_lpc.astype(np.float32)).clone()
    # _f_resid = torch.from_numpy(_f_resid.astype(np.float32)).clone()
    # gain = torch.from_numpy(gain.astype(np.float32)).clone()

    _wav = lpc_synthesis(_lpc, _f_resid, gain, config.hop_size)

    # -> numpy
    # _wav = _wav.to('cpu').detach().numpy().copy()

    scipy.io.wavfile.write(save_path, config.sampling_rate, _wav)

    _wav = (_wav * 32768).astype("int16")
    sys.exit(0)


if __name__ == "__main__":
    main()
