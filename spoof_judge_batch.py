#!/usr/bin/env python3
"""
Script to judge if an audio file is spoofed or bonafide using AASIST model.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license


python spoof_judge.py --audio_path ./mydata/test.wav --model_path models/weights/AASIST.pth --config config/AASIST.conf

python spoof_judge.py --audio_path ./mydata/test.wav --model_path models/weights/AASIST.pth --config config/AASIST.conf --atk_amp 0.1 --atk_f 250.1
"""
import argparse
import re
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
from numba.core.types import Boolean
from tqdm import tqdm

from spoof_judge import judge_spoof
from contextlib import redirect_stdout

warnings.filterwarnings("ignore", category=FutureWarning)
SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Judge if an audio file is spoofed or bonafide")
    parser.add_argument("--audio_path",
                        dest="audio_path",
                        type=str,
                        help="path to audio file to judge",
                        default="mydata/VoxCeleb1/attacker_audio/")
    parser.add_argument("--model_path",
                        dest="model_path",
                        type=str,
                        help="path to model weights",
                        default="./models/weights/AASIST.pth")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default="./config/AASIST.conf")
    parser.add_argument("--device",
                        dest="device",
                        type=str,
                        help="device to use (cuda or cpu)",
                        default=None)
    parser.add_argument("--atk",
                        dest="atk",
                        type=bool,
                        help="attack or not",
                        default=True)


    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.audio_path):
        print("Error: Audio file not found: {}".format(args.audio_path))
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print("Error: Model file not found: {}".format(args.model_path))
        sys.exit(1)

    if not os.path.exists(args.config):
        print("Error: Config file not found: {}".format(args.config))
        sys.exit(1)

    atk_amps =[]
    atk_fs=[]
    if args.atk:
        print("Attack")
        if "aishell" in args.audio_path:
            atk_amps=[0.0581, 0.0648, 0.036, 0.2922, 0.1546, 0.0095, 0.0573, 0.0555, 0.0436, 0.3988]
            atk_fs=[3671.06,4592.98,943.95,3542.28,4954.2,2133,636.12,1440.66,332.77,696.97]
        if "VoxCeleb" in args.audio_path:
            atk_amps=[0.5,0.5,0.3966,0.1178,0.44,0.5,0.5,0.3378,0.5,0.1344]
            atk_fs=[1999.99,10000,7060.15,6583.37,9498.15,3347.5,3100.75,4320.05,5000,1074.48]


    spoof_probs = []
    spoof_flags = []
    # 遍历audio_path下的所有文件
    for file in os.listdir(args.audio_path):
        if not file.endswith(".wav"):
            continue
        # 提取文件名中的数字
        match = re.search(r'\d+', file)
        if match:
            number = int(match.group())
            print(f"File: {file}, Number: {number}")
        else:
            print(f"File: {file}, No number found")

        # 循环100次，记录平均Spoof probability
        for i in tqdm(range(100)):
            with redirect_stdout(open(os.devnull, 'w')):
                if args.atk:
                    is_spoof, confidence, spoof_prob, bonafide_prob = judge_spoof(
                        args.audio_path + file, args.model_path, args.config, args.device, atk_amps[number-1], atk_fs[number-1])
                else:
                    is_spoof, confidence, spoof_prob, bonafide_prob = judge_spoof(
                        args.audio_path + file, args.model_path, args.config, args.device, None, None)
                spoof_probs.append(spoof_prob)
                spoof_flags.append(1 if is_spoof else 0)
    print("Average Spoof probability: {:.4f}".format(np.mean(spoof_probs)))
    print("Average Spoof percent: {:.4f}".format(np.sum(spoof_flags) / len(spoof_flags)))

if __name__ == "__main__":
    main()