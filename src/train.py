import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from attrdict import AttrDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio2mel import Audio2Mel
from data import AudioDataset
from model import MultiPeriodDiscriminator, MultiScaleDiscriminator, NeuralCELP
from utils import calc_pesq, calc_world_metrics, rc_synthesis

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def files_to_list(filename):
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def save_checkpoint(filepath, netE, netDs, netDp, optE, optD, step, config):
    state = {
        "config": config,
        "step": step,
        "netE": netE.state_dict(),
        "netDs": netDs.state_dict(),
        "netDp": netDp.state_dict(),
        "optE": optE.state_dict(),
        "optD": optD.state_dict(),
    }

    torch.save(state, filepath)


def read_checkpoint(filepath, netE, netDs, netDp, optE, optD, load_opt):
    state = torch.load(filepath)
    netE.load_state_dict(state["netE"])
    netDs.load_state_dict(state["netDs"])
    netDp.load_state_dict(state["netDp"])
    if load_opt:
        optE.load_state_dict(state["optE"])
        optD.load_state_dict(state["optD"])
        step = state["step"]
    else:
        step = 0

    return step


def train_step(netE, netDs, netDp, optE, optD, fft, train_loader, epoch, step, writer, config):
    netE.train()
    netDs.train()
    netDp.train()

    running_loss_adv = 0
    running_loss_fm = 0
    running_loss_g = 0
    running_loss_mel = 0
    running_loss_adv_d = 0
    running_loss_d = 0
    c = 0

    for x, rc_feature, z, gain in train_loader:
        x = x.cuda(non_blocking=True)
        rc_feature = rc_feature.cuda(non_blocking=True)
        z = z.cuda(non_blocking=True)
        gain = gain.cuda(non_blocking=True)

        with torch.no_grad():
            s = fft(x)

        z_hat = netE(z)

        x_reconst = rc_synthesis(
            rc_feature,
            z_hat,
            gain,
            config.frame_size,
            config.hop_size,
        )

        c += 1

        # for debug
        s_reconst = fft(x_reconst)

        ds_fake_det, _ = netDs(x_reconst.detach())
        ds_real, _ = netDs(x)

        dp_fake_det, _ = netDp(x_reconst.detach())
        dp_real, _ = netDp(x)

        loss_adv_ds = 0
        for v in ds_fake_det:
            loss_adv_ds += torch.mean(v**2)

        for v in ds_real:
            loss_adv_ds += torch.mean((1 - v)**2)

        loss_adv_dp = 0
        for v in dp_fake_det:
            loss_adv_dp += torch.mean(v**2)

        for v in dp_real:
            loss_adv_dp += torch.mean((1 - v)**2)

        loss_adv_d = loss_adv_ds + loss_adv_dp
        loss_d = loss_adv_d

        optD.zero_grad()
        loss_d.backward()
        optD.step()

        running_loss_adv_d += loss_adv_d.item()
        running_loss_d += loss_d.item()

        ds_fake, ds_fake_fm = netDs(x_reconst)
        _, ds_real_fm = netDs(x)

        dp_fake, dp_fake_fm = netDp(x_reconst)
        _, dp_real_fm = netDp(x)

        loss_adv = 0
        for v in ds_fake:
            loss_adv += torch.mean((1 - v)**2)

        for v in dp_fake:
            loss_adv += torch.mean((1 - v)**2)

        loss_fm = 0
        for fake, real in zip(ds_fake_fm, ds_real_fm):
            for fa, re in zip(fake, real):
                loss_fm += F.l1_loss(fa, re)

        for fake, real in zip(dp_fake_fm, dp_real_fm):
            for fa, re in zip(fake, real):
                loss_fm += F.l1_loss(fa, re)

        loss_mel = F.l1_loss(s_reconst, s)
        loss = (config.lambda_adv * loss_adv + config.lambda_fm * loss_fm +
                config.lambda_mel * loss_mel)

        loss = loss_mel

        optE.zero_grad()
        loss.backward()
        optE.step()

        running_loss_g += loss.item()
        running_loss_adv += loss_adv.item()
        running_loss_fm += loss_fm.item()
        running_loss_mel += loss_mel.item()

        step += 1

    divide = len(train_loader)
    if epoch % config.log_interval == 0 or epoch == 1:
        writer.add_scalar("loss/generator", running_loss_g / divide, epoch)
        writer.add_scalar("loss/g_adv_loss", running_loss_adv / divide, epoch)
        writer.add_scalar("loss/fm_loss", running_loss_fm / divide, epoch)
        writer.add_scalar("loss/mel_loss", running_loss_mel / divide, epoch)

        writer.add_scalar("loss/d_adv_loss", running_loss_adv_d / divide, epoch)
        writer.add_scalar("loss/discriminator", running_loss_d / divide, epoch)

    return step


def test_generate(netE, eval_loader, epoch, writer, config):
    netE.eval()

    with torch.no_grad():
        pesq = 0
        rmse_f0 = 0
        mcd = 0
        for i, (x, rc_feature, z, gain) in enumerate(eval_loader):
            x = x.cuda()
            rc_feature = rc_feature.cuda()
            # rc_feature = rc_feature.to('cpu').detach().numpy().copy()
            z = z.cuda()
            gain = gain.cuda()
            z_hat = netE(z)

            x_reconst = rc_synthesis(
                rc_feature,
                z_hat,
                gain,
                config.frame_size,
                config.hop_size,
            )

            x = x.squeeze().cpu().numpy()
            x_reconst = x_reconst.squeeze().cpu()
            # x_reconst = x_reconst.numpy()

            # TODO: 応急処置の為削除する。生成音声のサンプル数が元と同じになるようにpadする。
            pad = len(x) - len(x_reconst)
            x_reconst = F.pad(x_reconst, (pad // 2, pad // 2), "constant", 0)

            pesq += calc_pesq(x, x_reconst.numpy(), config.sampling_rate)
            r, m = calc_world_metrics(x, x_reconst.numpy(), config.sampling_rate)
            rmse_f0 += r
            mcd += m

            if epoch % config.eval_interval == 0 and i < config.n_test_samples:
                if writer is not None:
                    x_reconst[x_reconst != x_reconst] = 0.0
                    writer.add_audio(
                        f"reconstruct/sample_{i}.wav",
                        x_reconst,
                        epoch,
                        sample_rate=config.sampling_rate,
                    )

        if writer is not None:
            divide = len(eval_loader)
            writer.add_scalar("metric/pesq", pesq / divide, epoch)
            writer.add_scalar("metric/rmse_f0", rmse_f0 / divide, epoch)
            writer.add_scalar("metric/mcd", mcd / divide, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("data_root", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--load_opt", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = AttrDict(json.load(f))

    os.makedirs(args.save_dir, exist_ok=True)

    log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    weight_dir = os.path.join(args.save_dir, "weight")
    os.makedirs(weight_dir, exist_ok=True)

    netE = NeuralCELP(config.codebook_size, config.hidden_channels, config.window_length).cuda()
    netDs = MultiScaleDiscriminator(config).cuda()
    netDp = MultiPeriodDiscriminator(config).cuda()

    optE = torch.optim.AdamW(netE.parameters(), lr=config.learning_rate_g, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(itertools.chain(netDs.parameters(), netDp.parameters()),
                             lr=config.learning_rate_d,
                             betas=(0.8, 0.99))

    if args.load_path is not None:
        step = read_checkpoint(args.load_path, netE, netDs, netDp, optE, optD, args.load_opt)
    else:
        step = 0

    files = files_to_list(os.path.join(args.data_root, "train_files.txt"))

    train_files = files[:-config.num_valid_files]
    valid_files = files[-config.num_valid_files:]

    fft = Audio2Mel(n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    win_length=config.win_length,
                    sampling_rate=config.sampling_rate,
                    n_mel_channels=config.n_mel_channels,
                    mel_fmin=config.mel_fmin,
                    mel_fmax=config.mel_fmax).cuda()

    tmp_set = AudioDataset(
        args.data_root,
        valid_files[:config.n_test_samples],
        config,
        test_loader=True,
    )
    tmp_loader = DataLoader(tmp_set, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (x, _, _, _) in enumerate(tmp_loader):
            writer.add_audio(
                f"original/sample_{i}.wav",
                x.squeeze(),
                0,
                sample_rate=config.sampling_rate,
            )

    del tmp_set
    del tmp_loader

    train_set = AudioDataset(
        args.data_root,
        train_files,
        config,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    eval_set = AudioDataset(
        args.data_root,
        valid_files,
        config,
        test_loader=True,
    )
    eval_loader = DataLoader(
        eval_set,
        shuffle=False,
        batch_size=1,
    )

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    print("check eval step")
    # test_generate(netE, eval_loader, 0, None, config)
    test_generate(netE, eval_loader, 0, writer, config)

    for epoch in range(1, config.num_epoch + 1):
        step = train_step(
            netE,
            netDs,
            netDp,
            optE,
            optD,
            fft,
            train_loader,
            epoch,
            step,
            writer,
            config,
        )

        if epoch % config.save_interval == 0:
            save_path = os.path.join(weight_dir, f"HiFiGAN_{epoch}.pt")
            save_checkpoint(save_path, netE, netDs, netDp, optE, optD, step, config)
        else:
            save_path = os.path.join(weight_dir, "HiFiGAN_latest.pt")
            save_checkpoint(save_path, netE, netDs, netDp, optE, optD, step, config)

        if epoch % config.eval_interval == 0:
            test_generate(netE, eval_loader, epoch, writer, config)

    save_path = os.path.join(weight_dir, f"HiFiGAN_{epoch}.pt")
    save_checkpoint(save_path, netE, netDs, netDp, optE, optD, step, config)


if __name__ == "__main__":
    main()
