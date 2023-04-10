import argparse
import os
import shutil

replaces = {
    "jvs058/VOICEACTRESS100_021": "VOICEACTRESS100_022.wav",
    "jvs058/VOICEACTRESS100_020": "VOICEACTRESS100_021.wav",
    "jvs058/VOICEACTRESS100_019": "VOICEACTRESS100_020.wav",
    "jvs058/VOICEACTRESS100_018": "VOICEACTRESS100_019.wav",
    "jvs058/VOICEACTRESS100_017": "VOICEACTRESS100_018.wav",
    "jvs058/VOICEACTRESS100_016": "VOICEACTRESS100_017.wav",
    "jvs058/VOICEACTRESS100_015": "VOICEACTRESS100_016.wav"
}

excludes = set([
    "jvs009/VOICEACTRESS100_086", "jvs009/VOICEACTRESS100_095", "jvs017/VOICEACTRESS100_082",
    "jvs018/VOICEACTRESS100_072", "jvs022/VOICEACTRESS100_047", "jvs024/VOICEACTRESS100_088",
    "jvs036/VOICEACTRESS100_057", "jvs038/VOICEACTRESS100_006", "jvs038/VOICEACTRESS100_041",
    "jvs043/VOICEACTRESS100_085", "jvs047/VOICEACTRESS100_085", "jvs048/VOICEACTRESS100_043",
    "jvs048/VOICEACTRESS100_076", "jvs051/VOICEACTRESS100_025", "jvs055/VOICEACTRESS100_056",
    "jvs055/VOICEACTRESS100_076", "jvs055/VOICEACTRESS100_099", "jvs058/VOICEACTRESS100_014",
    "jvs059/VOICEACTRESS100_061", "jvs059/VOICEACTRESS100_064", "jvs059/VOICEACTRESS100_066",
    "jvs059/VOICEACTRESS100_074", "jvs060/VOICEACTRESS100_082", "jvs074/VOICEACTRESS100_062",
    "jvs098/VOICEACTRESS100_060", "jvs098/VOICEACTRESS100_099"
])


def read_transcripts(file):
    with open(file, encoding="utf-8") as f:
        rows = f.readlines()

    rows = [row.rstrip().split(":", 1) for row in rows]

    return rows


def prepare_files(root_dir, wav_dir, text_dir):
    for speaker in sorted(os.listdir(root_dir)):
        speaker_dir = os.path.join(root_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        speaker_txt_dir = os.path.join(text_dir, speaker)
        os.makedirs(speaker_txt_dir, exist_ok=True)
        speaker_wav_dir = os.path.join(wav_dir, speaker)
        os.makedirs(speaker_wav_dir, exist_ok=True)

        for subset in os.listdir(speaker_dir):
            if subset != "parallel100" and subset != "nonpara30":
                continue

            texts = read_transcripts(os.path.join(speaker_dir, subset, "transcripts_utf8.txt"))
            for file, text in texts:
                file_id = f"{speaker}/{file}"
                if file_id in excludes:
                    continue

                if file_id in replaces:
                    wav_filename = replaces[file_id]
                    wav_path = os.path.join(speaker_dir, subset, "wav24kHz16bit", wav_filename)
                else:
                    wav_path = os.path.join(speaker_dir, subset, "wav24kHz16bit", file + ".wav")

                if not os.path.exists(wav_path):
                    continue

                text_path = os.path.join(speaker_txt_dir, file + ".txt")
                save_wav_path = os.path.join(speaker_wav_dir, file + ".wav")

                with open(text_path, 'w') as f:
                    f.write(text.rstrip())

                shutil.copy(wav_path, save_wav_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jvs_root', type=str)
    parser.add_argument('data_root', type=str)
    args = parser.parse_args()

    wav_dir = os.path.join(args.data_root, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    text_dir = os.path.join(args.data_root, "txt")
    os.makedirs(text_dir, exist_ok=True)

    prepare_files(args.jvs_root, wav_dir, text_dir)


if __name__ == '__main__':
    main()
