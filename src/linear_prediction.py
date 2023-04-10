import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile
import torch


def analysis(wav, fl, fs, order, emph_coef=0.97, flag_emph=False):
    """lpc_coef, ld_err, gamma, gain, framed_err, err_signal = analysis(wav)
        
        LPC analysis on each frame
        
        input
        -----
          wav: np.array, (length, 1)
          
        output
        ------
          lpc_coef:   np.array, LPC coeff, (frame_num, lpc_order + 1)
          ld_err:     np.array, LD analysis error, (frame_num, lpc_order + 1)
          gamma:      np.array, reflection coefficients, (frame_num,lpc_order)
          gain:       np.array, gain, (frame_num, 1)
          framed_err: np.array, LPC error per frame, (frame_num, frame_length)
          eer_signal: np.array, overlap-added excitation (length, 1)

        Note that framed_err is the excitation signal from LPC analysis on each
        frame. eer_signal is the overlap-added excitation signal.
        """

    wav = np.expand_dims(wav, axis=1)
    if flag_emph:
        wav_tmp = preemphasis(wav)
    else:
        wav_tmp = wav

    # framing & windowing
    # win = np.hanning(fl)
    win = scipy.signal.windows.boxcar(fl)
    frame_wined = windowing(framing(wav_tmp[:, 0], fl, fs), win)

    # auto-correlation
    auto = auto_correlation(frame_wined, fl, order)

    # LD analysis
    lpc_coef, lpc_err, gamma_array, gain = levison_durbin(auto)

    # get LPC excitation signals in each frame
    framed_err = lpc_analysis_core(lpc_coef, frame_wined, gain)

    # overlap-add for excitation signal
    err_signal = overlapadd(framed_err, fl, fs, win)

    return lpc_coef, lpc_err, gamma_array, gain, framed_err, err_signal


def lpc_analysis_core(lpc_coef, framed_x, gain):
    """framed_err = _lpc_analysis_core(lpc_coef, framed_x, gain)
        
        LPC analysis on frame
        MA filtering: e[n] = \sum_k=0 a_k x[n-k] / gain
        
        input
        -----
          lpc_coef: np.array, (frame_num, order + 1)
          framed_x: np.array, (frame_num, frame_length)
          gain: np.array,     (frame_num, 1)
          
        output
        ------
          framed_err: np.array, (frame_num, frame_length)
        
        Note that lpc_coef[n, :] = (1, a_1, a_2, ..., a_order) for n-th frame
        framed_x[n, :] = (x[0], x[1], ..., x[frame_len]) for n-th frame
        """
    #
    frame_num = framed_x.shape[0]
    frame_len = framed_x.shape[1]

    # lpc order (without the a_0 term)
    order = lpc_coef.shape[1] - 1

    # pad zero, every frame has [0, ..., 0, x[0], x[1], ..., x[frame_len]]
    tmp_framed = np.concatenate([np.zeros([frame_num, order + 1]), framed_x], axis=1)

    # flip to (x[frame_len], ... x[1], x[0], 0, ..., 0)
    tmp_framed = tmp_framed[:, ::-1]

    # LPC excitation buffer
    framed_err = np.zeros_like(framed_x)

    # e[n] = \sum_k=0 a[k] x[n-k]
    # do this for all frames and n simultaneously
    for k in range(order + 1):
        # a[k]
        tmp_coef = lpc_coef[:, k:k + 1]

        # For each frame
        # RHS = [x[n-k], x[n-k-1], ..., ] * a[k]
        #
        # By doing this for k in [0, order]
        # LHS = [e[n],   e[n-1],   ...]
        #       [x[n-0], x[n-0-1], ..., ] * a[0]
        #     + [x[n-1], x[n-1-1], ..., ] * a[1]
        #     + [x[n-2], x[n-2-1], ..., ] * a[2]
        #     + ...
        # We get the excitation for one frame
        # This process is conducted for all frames at the same time
        framed_err += tmp_framed[:, 0:frame_len] * tmp_coef

        # roll to [x[n-k-1], x[n-k-2], ..., ]
        tmp_framed = np.roll(tmp_framed, -1, axis=1)

    # revese to (e[0], e[1], ..., e[frame_len])
    return framed_err[:, ::-1] / gain


def synthesis(lpc_coef, framed_err, gain, fl, fs, flag_emph=False):
    """wav = synthesis(lpc_coef, framed_err, gain):
    
    LPC synthesis (and overlap-add)
    
    input
    -----
        lpc_coef:   np.array, LPC coeff, (frame_num, lpc_order + 1)
        framed_err: np.array, LPC excitations, (frame_num, frame_length)
        gain:       np.array, LPC gain, (frame_num, 1)
    
    output
    ------
        wav: np.array, (length, 1)
    
    This function does LPC synthesis in each frame and create
    the output waveform by overlap-adding
    """
    # win = np.hanning(fl)
    win = scipy.signal.windows.boxcar(fl)
    framed_x = lpc_synthesis_core(lpc_coef, framed_err, gain)
    wav_tmp = overlapadd(framed_x, fl, fs, win)
    if flag_emph:
        wav_tmp = deemphasis(wav_tmp)
    return wav_tmp


def preemphasis(wav):
    """ wav_out = _preemphasis(wav)

    input
    -----
        wav: np.array, (length)

    output
    ------
        wav: np.array, (length)
    """
    wav_out = np.zeros_like(wav) + wav
    wav_out[1:] = wav_out[1:] - wav_out[0:-1] * emph_coef
    return wav_out


def deemphasis(wav):
    """ wav_out = _deemphasis(wav)

    input
    -----
        wav: np.array, (length)

    output
    ------
        wav: np.array, (length)
    """
    wav_out = np.zeros_like(wav) + wav
    for idx in range(1, wav.shape[0]):
        wav_out[idx] = wav_out[idx] + wav_out[idx - 1] * emph_coef
    return wav_out


def framing(wav, fl, fs):
    """F = _framed(wav)
    
    Framing the signal
    
    input
    -----
        wav: np.array, (length)

    output
    ------
        F: np.array, (frame_num, frame_length)
    """
    frame_num = (wav.shape[0] - fl) // fs + 1
    F = np.zeros([frame_num, fl], dtype=wav.dtype)
    for frame_idx in np.arange(frame_num):
        F[frame_idx, :] = wav[frame_idx * fs:frame_idx * fs + fl]
    return F


def windowing(framed_x, win):
    """windowing
    """
    return framed_x * win


def overlapadd(framed_x, fl, fs, win):
    """wav = _overlapadd(framed_x)
    
    Do overlap-add on framed (and windowed) signal
    
    input
    -----
        framed_x: np.array, (frame_num, frame_length)
        
    output
    ------
        wav: np.array, (length, 1)
    
    length = (frame_num - 1) * frame_shift + frame_length
    """
    # waveform length
    wavlen = (framed_x.shape[0] - 1) * fs + fl
    wavbuf = np.zeros([wavlen])

    # buf to save overlapped windows (to normalize the signal amplitude)
    protobuf = np.zeros([wavlen])
    win_prototype = windowing(framing(np.ones_like(protobuf), fl, fs), win)

    # overlap and add
    for idx in range(framed_x.shape[0]):
        frame_s = idx * fs
        wavbuf[frame_s:frame_s + fl] += framed_x[idx]
        protobuf[frame_s:frame_s + fl] += win_prototype[idx]

    # remove the impact of overlapped windows
    #protobuf[protobuf<1e-05] = 1.0
    wavbuf = wavbuf / protobuf.mean()
    return np.expand_dims(wavbuf, axis=1)


def lpc_analysis_core(lpc_coef, framed_x, gain):
    """framed_err = _lpc_analysis_core(lpc_coef, framed_x, gain)
    
    LPC analysis on frame
    MA filtering: e[n] = \sum_k=0 a_k x[n-k] / gain
    
    input
    -----
        lpc_coef: np.array, (frame_num, order + 1)
        framed_x: np.array, (frame_num, frame_length)
        gain: np.array,     (frame_num, 1)
        
    output
    ------
        framed_err: np.array, (frame_num, frame_length)
    
    Note that lpc_coef[n, :] = (1, a_1, a_2, ..., a_order) for n-th frame
    framed_x[n, :] = (x[0], x[1], ..., x[frame_len]) for n-th frame
    """
    #
    frame_num = framed_x.shape[0]
    frame_len = framed_x.shape[1]

    # lpc order (without the a_0 term)
    order = lpc_coef.shape[1] - 1

    # pad zero, every frame has [0, ..., 0, x[0], x[1], ..., x[frame_len]]
    tmp_framed = np.concatenate([np.zeros([frame_num, order + 1]), framed_x], axis=1)

    # flip to (x[frame_len], ... x[1], x[0], 0, ..., 0)
    tmp_framed = tmp_framed[:, ::-1]

    # LPC excitation buffer
    framed_err = np.zeros_like(framed_x)

    # e[n] = \sum_k=0 a[k] x[n-k]
    # do this for all frames and n simultaneously
    for k in range(order + 1):
        # a[k]
        tmp_coef = lpc_coef[:, k:k + 1]

        # For each frame
        # RHS = [x[n-k], x[n-k-1], ..., ] * a[k]
        #
        # By doing this for k in [0, order]
        # LHS = [e[n],   e[n-1],   ...]
        #       [x[n-0], x[n-0-1], ..., ] * a[0]
        #     + [x[n-1], x[n-1-1], ..., ] * a[1]
        #     + [x[n-2], x[n-2-1], ..., ] * a[2]
        #     + ...
        # We get the excitation for one frame
        # This process is conducted for all frames at the same time
        framed_err += tmp_framed[:, 0:frame_len] * tmp_coef

        # roll to [x[n-k-1], x[n-k-2], ..., ]
        tmp_framed = np.roll(tmp_framed, -1, axis=1)

    # revese to (e[0], e[1], ..., e[frame_len])
    return framed_err[:, ::-1] / gain


def lpc_synthesis_core(lpc_coef, framed_err, gain):
    """framed_x = _lpc_synthesis_core(lpc_coef, framed_err, gain)
    
    AR filtering: x[n] = gain * e[n] - \sum_k=0 a_k x[n-k]
    LPC synthesis on frame
    
    input
    -----
        lpc_coef:   np.array, (frame_num, order + 1)
        framed_err: np.array, (frame_num, frame_length)
        gain:       np.array, (frame_num, 1)
        
    output
    ------
        framed_x:   np.array, (frame_num, frame_length)
    
    Note that 
    lpc_coef[n, :] = (1, a_1, a_2, ..., a_order),  for n-th frame
    framed_x[n, :] = (x[0], x[1], ..., x[frame_len]), for n-th frame
    """
    frame_num = framed_err.shape[0]
    frame_len = framed_err.shape[1]
    order = lpc_coef.shape[1] - 1

    # pad zero
    # the buffer looks like
    #  [[0, 0, 0, 0, 0, ... x[0], x[1], x[frame_length -1]],  -> 1st frame
    #   [0, 0, 0, 0, 0, ... x[0], x[1], x[frame_length -1]],  -> 2nd frame
    #   ...]
    framed_x = np.concatenate([np.zeros([frame_num, order]), np.zeros_like(framed_err)], axis=1)

    # flip the cofficients of each frame as [a_order, ..., a_1, 1]
    lpc_coef_tmp = lpc_coef[:, ::-1]

    # synthesis (all frames are down at the same time)
    for idx in range(frame_len):
        # idx+order so that it points to the shifted time idx
        #                                    idx+order
        # [0, 0, 0, 0, 0, ... x[0], x[1], ... x[idx], ]
        # gain * e[n]
        framed_x[:, idx + order] = framed_err[:, idx] * gain[:, 0]

        # [x[idx-1-order], ..., x[idx-1]] * [a_order, a_1]
        pred = np.sum(framed_x[:, idx:idx + order] * lpc_coef_tmp[:, :-1], axis=1)

        # gain * e[n] - [x[idx-1-order], ..., x[idx-1]] * [a_order, a_1]
        framed_x[:, idx + order] = framed_x[:, idx + order] - pred

    # [0, 0, 0, 0, 0, ... x[0], x[1], ... ] -> [x[0], x[1], ...]
    return framed_x[:, order:]


def auto_correlation(framed_x, fl, order):
    """ autocorr = _auto_correlation(framed_x)

    input
    -----
        framed_x: np.array, (frame_num, frame_length), frame windowed signal
    
    output
    ------
        autocorr: np.array, auto-correlation coeff (frame_num, lpc_order+1)
    """
    # (frame_num, order)
    autocor = np.zeros([framed_x.shape[0], order + 1])

    # loop and compute auto-corr (for all frames simultaneously)
    for i in np.arange(order + 1):
        autocor[:, i] = np.sum(framed_x[:, 0:fl - i] * framed_x[:, i:], axis=1)
    # (frame_num, order)
    return autocor


def levison_durbin(autocor):
    """lpc_coef_ou, lpc_err, gamma_array, gain = _levison_durbin(autocor)
    Levison durbin
    
    input
    -----
        autocor: np.array, auto-correlation, (frame_num, lpc_order+1)
    
    output
    ------
        lpc_coef: np.array, LPC coefficients, (frame_num, lpc_order+1)
        lpc_err: np.array, LPC error, (frame_num, lpc_order+1)
        gamma: np.array, reflection coeff, (frame_num, lpc_order)
        gain: np.array, gain, (frame_num, 1)
        
    Note that lpc_coef[n] = (1, a_2, ... a_order) for n-th frame
    """
    # (frame_num, order)
    frame_num, order = autocor.shape
    order = order - 1
    polyOrder = order + 1

    # to log down the invalid frames
    tmp_order = np.zeros([frame_num], dtype=np.int32) + polyOrder

    lpc_coef = np.zeros([frame_num, 2, polyOrder])
    lpc_err = np.zeros([frame_num, polyOrder])
    gamma_array = np.zeros([frame_num, order])
    gain = np.zeros([frame_num])

    lpc_err[:, 0] = autocor[:, 0]
    lpc_coef[:, 0, 0] = 1.0

    for index in np.arange(1, polyOrder):

        lpc_coef[:, 1, index] = 1.0

        # compute gamma
        #   step1.
        gamma = np.sum(lpc_coef[:, 0, 0:(index)] * autocor[:, 1:(index + 1)], axis=1)
        #   step2. check validity of lpc_err
        ill_idx = lpc_err[:, index - 1] < 1e-07
        #      also frames that should have been stopped in previous iter
        ill_idx = np.bitwise_or(ill_idx, tmp_order < polyOrder)
        #   step3. make invalid frame gamma=0
        gamma[ill_idx] = 0
        gamma[~ill_idx] = gamma[~ill_idx] / lpc_err[~ill_idx, index - 1]
        gamma_array[:, index - 1] = gamma
        #   step4. log down the ill frames
        tmp_order[ill_idx] = index

        lpc_coef[:, 1, 0] = -1.0 * gamma
        if index > 1:
            lpc_coef[:, 1, 1:index] = lpc_coef[:, 0, 0:index-1] \
            + lpc_coef[:, 1, 0:1] * lpc_coef[:, 0, 0:index-1][:, ::-1]

        lpc_err[:, index] = lpc_err[:, index - 1] * (1 - gamma * gamma)
        lpc_coef[:, 0, :] = lpc_coef[:, 1, :]

    # flip to (1, a_1, ..., a_order)
    lpc_coef = lpc_coef[:, 0, ::-1]

    # output LPC coefficients
    lpc_coef_ou = np.zeros([frame_num, polyOrder])

    # if high-order LPC analysis is not working
    # each frame may require a different truncation length
    for idx in range(frame_num):
        lpc_coef_ou[idx, 0:tmp_order[idx]] = lpc_coef[idx, 0:tmp_order[idx]]

    # get the gain, when tmp_order = polyOrder, tmp_order-2 -> order-1,
    #  last element of the lpc_err buffer
    gain = np.sqrt(lpc_err[np.arange(len(tmp_order)), tmp_order - 2])

    # if the gain is zero, it means analysis error is zero,
    gain[gain < 1e-07] = 1.0

    # (frame_num, order)
    return lpc_coef_ou, lpc_err, gamma_array, np.expand_dims(gain, axis=1)


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

    lpc, lpc_err, gamma, gain, f_resid, err_signal = analysis(wav,
                                                              config.frame_size,
                                                              config.hop_size,
                                                              order=config.lpc_order)
    _wav = synthesis(lpc, f_resid, gain, config.frame_size, config.hop_size)

    # numpy->torch
    # _lpc = torch.from_numpy(_lpc.astype(np.float32)).clone()
    # _f_resid = torch.from_numpy(_f_resid.astype(np.float32)).clone()
    # gain = torch.from_numpy(gain.astype(np.float32)).clone()

    scipy.io.wavfile.write(save_path, config.sampling_rate, _wav)

    _wav = (_wav * 32768).astype("int16")
    sys.exit(0)


if __name__ == "__main__":
    main()
