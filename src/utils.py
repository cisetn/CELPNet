import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
from pesq import pesq

from lpc_gpu import (framing_gpu, lpc_analysis, lpc_synthesis, overlapadd, rc2lpc)

# def calc_lsp(x, lpc_order, frame_size, hop_size):
#     lpc_feature, framed_resid, gain, _ = lpc_analysis(x, lpc_order, frame_size, hop_size)
#     z = overlappadd(framed_resid, frame_size, hop_size)

#     lsp_feature = lpc_to_lsp(lpc_feature)
#     return lsp_feature, z, gain

# def lsp_synthesis(lsp_feature, z, gain, hop_size):
#     lpc_feature = lsp_to_lpc(lsp_feature)
#     framed_resid = framing(z)
#     x = lpc_synthesis(lpc_feature, framed_resid, gain, hop_size)
#     return x


def calc_rc(wav, lpc_order, frame_size, hop_size):
    # lpc_feature, frame_resid, gain, rc_feature, z = lpc_analysis(x, lpc_order, frame_size, hop_size)
    lpc, ld_err, rc, gain, f_resid, resid = lpc_analysis(wav, frame_size, hop_size, lpc_order)

    return rc, resid, gain


def rc_synthesis(rc_feature, z, gain, frame_size, hop_size):
    _lpc = rc2lpc(rc_feature)
    _f_resid = framing_gpu(z, frame_size, hop_size)
    # win = get_window(window, frame_size)
    # _f_resid = windowing(_f_resid, win)
    _wav = lpc_synthesis(_lpc, _f_resid, gain, hop_size)

    return _wav


def calc_pesq(ref, deg, orig_sr):
    ref = librosa.core.resample(ref, orig_sr=orig_sr, target_sr=16000)
    deg = librosa.core.resample(deg, orig_sr=orig_sr, target_sr=16000)

    return pesq(16000, ref, deg, 'wb')


def extract_world_features(wav, sr, dim=25):
    f0, _time = pw.harvest(wav.astype(np.float64), fs=sr, frame_period=5.0)
    f0 = pw.stonemask(wav.astype(np.float64), f0, _time, fs=sr)
    sp = pw.cheaptrick(wav.astype(np.float64), f0, _time, fs=sr)
    mcep = pw.code_spectral_envelope(sp, sr, dim)

    return f0, mcep


def calc_rmse(x, y):
    f0_r_uv = (x == 0) * 1
    f0_r_v = 1 - f0_r_uv
    f0_s_uv = (y == 0) * 1
    f0_s_v = 1 - f0_s_uv

    tp_mask = f0_r_v * f0_s_v

    tmp = 1200 * np.abs(np.log2(x + f0_r_uv) - np.log2(y + f0_s_uv))
    tmp = tmp * tp_mask
    rmse_f0_mean = tmp.sum() / tp_mask.sum()

    return rmse_f0_mean


def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    diff = x - y

    return log_spec_dB_const * np.sqrt(np.inner(diff, diff))


def calc_mcd(x, y):
    mcd = np.mean(10 / np.log(10) * np.sqrt(2 * np.sum((x[:, 1:] - y[:, 1:])**2, axis=1)))

    return mcd


def calc_world_metrics(x, y, sr):
    f0_x, mcep_x = extract_world_features(x, sr)
    f0_y, mcep_y = extract_world_features(y, sr)

    rmse_f0 = calc_rmse(f0_x, f0_y)
    mcd = calc_mcd(mcep_x, mcep_y)

    return rmse_f0, mcd
