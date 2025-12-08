#!/usr/bin/env python3
"""
Script to judge if an audio file is spoofed or bonafide using AASIST model.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license


python swap_atk_param.py --audio_path ./mydata/voxceleb/pair1.wav --model_path models/weights/AASIST.pth --config config/AASIST.conf --amp_range 0.0 0.1 20 --f_range 0.0 20000.0 41

python swap_atk_param.py --model_path models/weights/AASIST.pth --config config/AASIST.conf --amp_range 0.0 0.5 26 --f_range 0.0 20000.0 26 --audio_path ./mydata/voxceleb/pair1/00005.wav

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
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
from spoof_judge import get_model, load_audio

warnings.filterwarnings("ignore", category=FutureWarning)
SAMPLE_RATE = 16000

# Global variable to cache the model and device
cache_model = None
cache_device = None


def judge_spoof(audio_path, model_path, config_path, device, atk_amp = None, atk_f = None):
    """Judge if an audio file is spoofed or bonafide"""
    global cache_model, cache_device
    
    # Initialize device if not cached
    if cache_device is None:
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        cache_device = device
        print("Device: {}".format(device))
        if device == "cpu":
            print("Warning: Using CPU, this will be slow")
    else:
        device = cache_device

    # Initialize model if not cached
    if cache_model is None:
        # Load configuration
        with open(config_path, "r") as f_json:
            config = json.loads(f_json.read())
        model_config = config["model_config"]

        # Define model architecture
        model = get_model(model_config, "cpu")  # Initialize on CPU first

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Model loaded : {}".format(model_path))
        
        # Move model to target device
        if device == "mps":
            # For MPS, we need to be more careful with device placement
            try:
                model = model.to(torch.device(device))
            except Exception as e:
                print(f"Failed to move model to MPS device: {e}")
                print("Falling back to CPU")
                device = "cpu"
                cache_device = "cpu"
                model = model.to(torch.device("cpu"))
        else:
            model = model.to(torch.device(device))
            
        cache_model = model
    else:
        model = cache_model
        # Ensure device consistency
        if device != cache_device:
            device = cache_device

    # Load audio (without showing plots during sweep)
    audio_tensor = load_audio(audio_path, atk_amp=atk_amp, atk_f=atk_f, show_plot=False)
    
    # Move tensor to device
    try:
        audio_tensor = audio_tensor.to(torch.device(device))
    except Exception as e:
        print(f"Failed to move tensor to device {device}: {e}")
        if device == "mps":
            print("Falling back to CPU for this iteration")
            device = "cpu"
            audio_tensor = audio_tensor.to(torch.device("cpu"))

    # Evaluate
    model.eval()
    with torch.no_grad():
        _, output = model(audio_tensor)
        # Get probabilities
        probabilities = torch.softmax(output, dim=1)
        spoof_prob = probabilities[0][0].item()  # Probability of spoof 从训练代码中验证过
        bonafide_prob = probabilities[0][1].item()  # Probability of bonafide

        # Determine result
        is_spoof = spoof_prob > bonafide_prob
        confidence = max(spoof_prob, bonafide_prob)

    return is_spoof, confidence, spoof_prob, bonafide_prob


def plot_spoof_heatmap(audio_path, model_path, config_path, device, amp_range, f_range):
    """Plot 2D heatmap of spoof probability across different attack amplitudes and frequencies"""
    # Create meshgrid for amp and f values
    amp_values = np.linspace(amp_range[0], amp_range[1], amp_range[2])
    f_values = np.linspace(f_range[0], f_range[1], f_range[2])
    amp_grid, f_grid = np.meshgrid(amp_values, f_values)
    
    # Initialize spoof probability grid
    spoof_prob_grid = np.zeros_like(amp_grid)

    _, _, base_spoof_prob, _ = judge_spoof(
        audio_path, model_path, config_path, device)
    print(f"Base spoof probability: {base_spoof_prob:.3f}")
    # Calculate spoof probabilities for each combination
    for i in range(len(f_values)):
        for j in range(len(amp_values)):
            _, _, spoof_prob, _ = judge_spoof(
                audio_path, model_path, config_path, device, amp_values[j], f_values[i])
            spoof_prob_grid[i, j] = spoof_prob - base_spoof_prob
            print(f"Processed: amp={amp_values[j]:.3f}, f={f_values[i]:.1f}, spoof_prob gap={spoof_prob_grid[i, j]:.3f}, spoof_prob={spoof_prob:.3f}")

    # 打印最大值、最小值、平均值
    print(f"Max spoof prob gap: {np.max(spoof_prob_grid):.3f}")
    print(f"Min spoof prob gap: {np.min(spoof_prob_grid):.3f}")
    print(f"Mean spoof prob gap: {np.mean(spoof_prob_grid):.3f}")

    # Plot heatmap
    plt.figure(figsize=(4, 3), constrained_layout=True)
    im = plt.imshow(spoof_prob_grid, cmap='RdYlBu', interpolation='bilinear',
                    extent=[amp_range[0], amp_range[1], f_range[0], f_range[1]], 
                    aspect='auto', origin='lower') # magma, viridis，coolwarm， RdYlBu
    plt.colorbar(im, label='概率差值')
    plt.xlabel('攻击幅度')
    plt.ylabel('攻击频率(Hz)')
    # plt.title('攻击后语音通过欺骗检测的概率相比于未攻击增加了多少（越小越好）')
    # plt.tight_layout()
    plt.show()
    
    return amp_grid, f_grid, spoof_prob_grid


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
    parser.add_argument("--amp_range",
                        dest="amp_range",
                        type=float,
                        nargs=3,
                        help="amplitude range as start end steps",
                        default=[0.0, 0.1, 10])
    parser.add_argument("--f_range",
                        dest="f_range",
                        type=float,
                        nargs=3,
                        help="frequency range as start end steps",
                        default=[0.0, 4000.0, 10])

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

    # If specific attack parameters are given, use the original functionality
    if args.atk_amp is not None and args.atk_f is not None:
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
    else:
        # Plot heatmap with ranges
        amp_range = (args.amp_range[0], args.amp_range[1], int(args.amp_range[2]))
        f_range = (args.f_range[0], args.f_range[1], int(args.f_range[2]))
        plot_spoof_heatmap(args.audio_path, args.model_path, args.config, args.device, amp_range, f_range)


if __name__ == "__main__":
    main()