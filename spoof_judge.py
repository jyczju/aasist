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
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import librosa
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
SAMPLE_RATE = 16000


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
        # # 起点从0～x_len-max_len之间进行取值，取值范围是0～x_len-max_len
        # stt = np.random.randint(x_len - max_len)
        # return x[stt:stt + max_len]

    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def get_model(model_config, device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model


def load_audio(audio_path, max_len=64600, atk_amp : float = None, atk_f :  float = None, show_plot=False):
    """Load audio file and pad/trim to max_len samples"""
    # Get file extension
    ext = os.path.splitext(audio_path)[1].lower()
    
    x, fs = sf.read(audio_path)
    
    if fs != SAMPLE_RATE:
        print("Sample rate is not 16kHz, resample to 16kHz")
        # Resample
        x = librosa.resample(x, orig_sr = fs, target_sr = SAMPLE_RATE)

    # Convert to mono if stereo (double-check)
    if len(x.shape) > 1:
        x = x[:, 0]  # Take first channel


    
    # Pad or trim to max_len
    x_pad = pad(x, max_len)

    # 对x进行归一化
    x_pad = x_pad / np.max(np.abs(x_pad))

    # 绘制音频波形图和时间频谱图在同一张图上
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        librosa.display.waveshow(x_pad, sr=SAMPLE_RATE, ax=ax1)
        ax1.set_title('Waveform')
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(x_pad)), ref=np.max),
                                 sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Spectrogram')
        plt.tight_layout()
        plt.show()

    # 如果是攻击，则在已有音频上叠加幅值为atk_amp、频率为atk_f的正弦波
    if atk_amp is not None and atk_f is not None:
        x_pad = x_pad + atk_amp * np.sin(2 * np.pi * atk_f * np.arange(x_pad.shape[0]) / 16000)

        if show_plot:
            # 绘制音频波形图和时间频谱图在同一张图上
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            librosa.display.waveshow(x_pad, sr=16000, ax=ax1)
            ax1.set_title('Waveform (with attack)')
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(x_pad)), ref=np.max),
                                     sr=16000, x_axis='time', y_axis='log', ax=ax2)
            ax2.set_title('Spectrogram (with attack)')
            plt.tight_layout()
            plt.show()

    x_tensor = Tensor(x_pad).unsqueeze(0)  # Add batch dimension
    return x_tensor


def judge_spoof(audio_path, model_path, config_path, device, atk_amp, atk_f):
    """Judge if an audio file is spoofed or bonafide"""
    # Load configuration
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]

    # Set device
    if device is None:
        if torch.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        print("Warning: Using CPU, this will be slow")

    # Define model architecture
    model = get_model(model_config, device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded : {}".format(model_path))
    
    # Load audio
    audio_tensor = load_audio(audio_path, atk_amp=atk_amp, atk_f=atk_f)
    audio_tensor = audio_tensor.to(device)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        _, output = model(audio_tensor)
        print("Output shape: {}".format(output.shape))
        print("Output: {}".format(output))
        # Get probabilities
        probabilities = torch.softmax(output, dim=1)
        spoof_prob = probabilities[0][0].item()  # Probability of spoof
        bonafide_prob = probabilities[0][1].item()  # Probability of bonafide
        
        # Determine result
        is_spoof = spoof_prob > bonafide_prob
        confidence = max(spoof_prob, bonafide_prob)
        
    return is_spoof, confidence, spoof_prob, bonafide_prob


def main():
    parser = argparse.ArgumentParser(description="Judge if an audio file is spoofed or bonafide")
    parser.add_argument("--audio_path",
                        dest="audio_path",
                        type=str,
                        help="path to audio file to judge",
                        required=True)
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
    parser.add_argument("--atk_amp",
                        dest="atk_amp",
                        type=float,
                        help="attack amplitude",
                        default=None)
    parser.add_argument("--atk_f",
                        dest="atk_f",
                        type=float,
                        help="attack frequency",
                        default=None)
                        
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
    
    # Judge spoof
    is_spoof, confidence, spoof_prob, bonafide_prob = judge_spoof(
        args.audio_path, args.model_path, args.config, args.device, args.atk_amp, args.atk_f)
    
    # Print results
    result = "SPOOF" if is_spoof else "BONAFIDE"
    print("\nResults:")
    print("========")
    print("File: {}".format(args.audio_path))
    print("Prediction: {}".format(result))
    print("Confidence: {:.2f}%".format(confidence * 100))
    print("Spoof probability: {:.2f}%".format(spoof_prob * 100))
    print("Bonafide probability: {:.2f}%".format(bonafide_prob * 100))


if __name__ == "__main__":
    main()