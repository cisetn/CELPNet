import argparse
import json
import os
import random

import librosa
import numpy as np
from attrdict import AttrDict
from tqdm import tqdm


def preprocess(save_dir, wav_path, config):
    wav, _ = librosa.core.load(wav_path, sr=config.sampling_rate)
    wav = librosa.util.normalize(wav) * 0.99
    wav, _ = librosa.effects.trim(wav, top_db=100, frame_length=2048, hop_length=512)
    wav = wav.astype(np.float32)

    np.save(os.path.join(save_dir, "wav.npy"), wav)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str)
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = AttrDict(json.load(f))

    feature_dir = os.path.join(args.data_root, "feature")
    os.makedirs(feature_dir, exist_ok=True)

    wav_dir = os.path.join(args.data_root, "wav")

    file_list = []
    for speaker in tqdm(os.listdir(wav_dir)):
        speaker_wav_dir = os.path.join(wav_dir, speaker)
        for wav_file in os.listdir(speaker_wav_dir):
            file_id = wav_file.replace(".wav", "")
            wav_path = os.path.join(speaker_wav_dir, wav_file)

            save_dir = os.path.join(feature_dir, speaker, file_id)
            os.makedirs(save_dir, exist_ok=True)

            result = preprocess(save_dir, wav_path, config)
            if result:
                file_list.append(f"{speaker}/{file_id}")

    random.seed(42)
    random.shuffle(file_list)
    train_files = file_list[:-config.num_test_files]
    test_files = file_list[-config.num_test_files:]

    train_list = os.path.join(args.data_root, "train_files.txt")
    with open(train_list, mode='w') as f:
        f.write('\n'.join(train_files))

    test_list = os.path.join(args.data_root, "test_files.txt")
    with open(test_list, mode='w') as f:
        f.write('\n'.join(test_files))


if __name__ == '__main__':
    main()
