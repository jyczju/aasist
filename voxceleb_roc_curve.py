#!/usr/bin/env python3
"""
VoxCeleb数据集上的活体检测ROC曲线绘制脚本

该脚本用于绘制活体检测模型在VoxCeleb数据集上的ROC曲线，
其中未攻击的音频定义为正样本，攻击后的音频定义为负样本。

Author: Lingma
"""

import argparse
import os
import sys
import warnings
import json
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch

from spoof_judge import judge_spoof

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 使用 macOS 自带的中文字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

warnings.filterwarnings("ignore", category=FutureWarning)
SAMPLE_RATE = 16000


class VoxCelebROCCalculator:
    """VoxCeleb数据集ROC曲线计算器"""
    
    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        
        # VoxCeleb攻击参数（根据spoof_judge_roc.py中的设置）
        self.atk_amps = [0.5, 0.5, 0.3966, 0.1178, 0.44, 0.5, 0.5, 0.3378, 0.5, 0.1344,
                        0.4641, 0.119, 0.481, 0.3819, 0.2124, 0.1794, 0.3569, 0.2895, 
                        0.3477, 0.4853]
        self.atk_fs = [1999.99, 10000, 7060.15, 6583.37, 9498.15, 3347.5, 3100.75, 
                      4320.05, 5000, 1074.48, 1468.86, 6159.21, 2667.74, 3018.91, 
                      618.74, 821.02, 3867.59, 1217.95, 614.54, 3976.73]
        
    def load_voxceleb_metadata(self, metadata_path: str) -> dict:
        """加载VoxCeleb元数据"""
        metadata = {}
        try:
            with open(metadata_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, filename, _, _, label = parts
                        metadata[filename] = label
        except FileNotFoundError:
            print(f"警告: 未找到元数据文件 {metadata_path}，将假设所有文件都是bonafide")
            metadata = {}
        return metadata
    
    def calculate_roc_points(self, audio_dir: str, metadata_path: str = None, 
                           iterations: int = 10, test_times_per_file: int = 5) -> Tuple[List[float], List[float], List[float]]:
        """
        计算ROC曲线的各个点
        
        Args:
            audio_dir: 音频文件目录
            metadata_path: 元数据文件路径（可选）
            iterations: 每次测试的迭代次数
            test_times_per_file: 每个音频文件的重复测试次数
            
        Returns:
            tuple: (y_true, y_scores_clean, y_scores_attacked)
                - y_true: 真实标签（1表示正样本/bonafide，0表示负样本/spoof）
                - y_scores_clean: 未攻击音频的bonafide概率分数
                - y_scores_attacked: 攻击后音频的bonafide概率分数
        """
        # 加载元数据
        metadata = self.load_voxceleb_metadata(metadata_path) if metadata_path else {}
        
        y_true = []          # 真实标签
        y_scores_clean = []  # 未攻击的bonafide概率
        y_scores_attacked = []  # 攻击后的bonafide概率
        
        # 获取所有wav文件
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print(f"找到 {len(wav_files)} 个音频文件")
        
        for file_idx, filename in enumerate(tqdm(wav_files, desc="Processing audio files")):
            file_path = os.path.join(audio_dir, filename)
            
            # 提取文件编号（用于匹配攻击参数）
            match = re.search(r'\d+', filename)
            attack_index = int(match.group()) - 1 if match else file_idx % len(self.atk_amps)
            
            # 确保索引在有效范围内
            attack_index = min(attack_index, len(self.atk_amps) - 1)
            
            # 构建标签：未攻击样本为正样本(1)，攻击样本为负样本(0)
            # 对于每个文件，我们生成test_times_per_file * iterations个未攻击样本和攻击样本
            total_samples_per_file = test_times_per_file * iterations
            y_true.extend([1] * total_samples_per_file)  # 未攻击样本标签为1（正样本）
            y_true.extend([0] * total_samples_per_file)  # 攻击样本标签为0（负样本）
            
            # 测试未攻击的情况（正样本）
            clean_bonafide_probs = []
            # 对每个文件进行多次测试以获得更多数据点
            for test_round in range(test_times_per_file):
                for _ in range(iterations):
                    try:
                        _, _, spoof_prob, bonafide_prob = judge_spoof(
                            file_path, self.model_path, self.config_path, self.device, None, None)
                        clean_bonafide_probs.append(bonafide_prob)
                    except Exception as e:
                        print(f"处理文件 {filename} 时出错: {e}")
                        clean_bonafide_probs.append(0.5)  # 出错时给中性分数
                        
            y_scores_clean.extend(clean_bonafide_probs)
            
            # 测试攻击后的情况（负样本）
            attacked_bonafide_probs = []
            amp = self.atk_amps[attack_index]
            freq = self.atk_fs[attack_index]
            
            # 对每个文件进行多次测试以获得更多数据点
            for test_round in range(test_times_per_file):
                for _ in range(iterations):
                    try:
                        _, _, spoof_prob, bonafide_prob = judge_spoof(
                            file_path, self.model_path, self.config_path, self.device, amp, freq)
                        attacked_bonafide_probs.append(bonafide_prob)
                    except Exception as e:
                        print(f"处理攻击文件 {filename} 时出错: {e}")
                        attacked_bonafide_probs.append(0.5)
                        
            y_scores_attacked.extend(attacked_bonafide_probs)
            
        return y_true, y_scores_clean, y_scores_attacked
    
    def plot_roc_curve(self, y_true: List[int], y_scores_clean: List[float], 
                      y_scores_attacked: List[float], save_path: str = None):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_scores_clean: 未攻击音频的分数
            y_scores_attacked: 攻击后音频的分数
            save_path: 保存路径（可选）
        """
        # 合并数据
        y_scores_all = y_scores_clean + y_scores_attacked
        # 注意：这里不需要重新构建y_pred_all，因为我们已经有了正确的y_true
        # y_true已经按照[正样本..., 负样本...]的顺序构建
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores_all)
        roc_auc = auc(fpr, tpr)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, marker='o', markersize=3,
                label=f'ROC curve (AUC = {roc_auc:.4f})', markevery=max(1, len(fpr)//20))
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier (AUC = 0.5)')
        
        # 设置图形属性
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (攻击被误判为正常的比例)', fontsize=12)
        plt.ylabel('True Positive Rate (正常样本被正确识别的比例)', fontsize=12)
        plt.title('活体检测模型在VoxCeleb数据集上的ROC曲线\n(未攻击=正样本, 攻击=负样本)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加AUC文本框
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}\n' 
                          f'测试样本数: {len(y_true)}\n'
                          f'正样本(未攻击): {len(y_scores_clean)}\n'
                          f'负样本(攻击): {len(y_scores_attacked)}',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                fontsize=10)
        
        # 移除重复的文本信息（已在上面添加）
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return roc_auc, fpr, tpr, thresholds
    
    def print_statistics(self, y_true: List[int], y_scores_clean: List[float], 
                        y_scores_attacked: List[float]):
        """打印统计信息"""
        print("\n=== ROC分析统计信息 ===")
        print(f"总样本数: {len(y_true)}")
        print(f"正样本数 (未攻击): {len(y_scores_clean)}")
        print(f"负样本数 (攻击): {len(y_scores_attacked)}")
        print(f"正样本平均bonafide概率: {np.mean(y_scores_clean):.4f} ± {np.std(y_scores_clean):.4f}")
        print(f"负样本平均bonafide概率: {np.mean(y_scores_attacked):.4f} ± {np.std(y_scores_attacked):.4f}")
        print(f"正样本最大bonafide概率: {np.max(y_scores_clean):.4f}")
        print(f"正样本最小bonafide概率: {np.min(y_scores_clean):.4f}")
        print(f"负样本最大bonafide概率: {np.max(y_scores_attacked):.4f}")
        print(f"负样本最小bonafide概率: {np.min(y_scores_attacked):.4f}")


def main():
    parser = argparse.ArgumentParser(description="绘制VoxCeleb数据集上的活体检测ROC曲线")
    parser.add_argument("--audio_dir",
                        dest="audio_dir",
                        type=str,
                        required=False,
                        help="VoxCeleb音频文件目录",
                        default="/Users/jiangyancheng/Library/CloudStorage/OneDrive-个人/Ghost-SV/evaluation_audio/merged/VoxCeleb1/target_audio/")
    parser.add_argument("--model_path",
                        dest="model_path",
                        type=str,
                        help="模型权重路径",
                        default="./models/weights/AASIST.pth")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="配置文件路径",
                        default="./config/AASIST.conf")
    parser.add_argument("--device",
                        dest="device",
                        type=str,
                        help="计算设备 (cuda/cpu/mps)",
                        default="mps")
    parser.add_argument("--metadata",
                        dest="metadata",
                        type=str,
                        help="元数据文件路径（可选）",
                        default=None)
    parser.add_argument("--test_times",
                        dest="test_times",
                        type=int,
                        help="每个音频文件的测试次数",
                        default=20)
    parser.add_argument("--save_path",
                        dest="save_path",
                        type=str,
                        help="ROC曲线保存路径",
                        default="./figure/voxceleb_roc_curve.png")
    
    args = parser.parse_args()
    
    # 检查必要文件是否存在
    if not os.path.exists(args.audio_dir):
        print(f"错误: 音频目录不存在: {args.audio_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建输出目录
    if args.save_path:
        output_dir = os.path.dirname(args.save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    print("开始计算VoxCeleb数据集上的ROC曲线...")
    print(f"音频目录: {args.audio_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print(f"每个文件测试次数: {args.test_times}")
    print(f"总测试点数: {len(os.listdir(args.audio_dir)) * args.test_times * 2}")
    
    # 初始化计算器
    calculator = VoxCelebROCCalculator(args.model_path, args.config, args.device)
    
    # 计算ROC点
    y_true, y_scores_clean, y_scores_attacked = calculator.calculate_roc_points(
        args.audio_dir, args.metadata, args.test_times)
    
    # 打印统计信息
    calculator.print_statistics(y_true, y_scores_clean, y_scores_attacked)
    
    # 绘制ROC曲线
    if len(y_true) > 0:
        auc_score, fpr, tpr, thresholds = calculator.plot_roc_curve(
            y_true, y_scores_clean, y_scores_attacked, args.save_path)
        print(f"\nROC AUC Score: {auc_score:.4f}")
    else:
        print("错误: 没有有效的测试数据")


if __name__ == "__main__":
    main()