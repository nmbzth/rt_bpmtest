#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time BPM detector using system playback (loopback) audio.

This script records the sound currently playing on your computer (not the
microphone) and estimates the tempo in near real time.

Dependencies:
    pip install soundcard numpy

Usage:
    python rt_bpm.py --list-devices
    python rt_bpm.py --device "Speakers"
    python rt_bpm.py
"""

import argparse
import sys
import time
from collections import deque

import numpy as np

try:
    import soundcard as sc
except ImportError:
    sys.exit(
        "缺少依赖 soundcard。请先安装：\n"
        "    pip install soundcard numpy\n"
    )


def list_devices():
    print("可用的输出设备（用于循环回录）：")
    for i, spk in enumerate(sc.all_speakers()):
        print(f"  [{i}] {spk.name}")


def select_speaker(device_arg):
    speakers = sc.all_speakers()
    if not speakers:
        sys.exit("没有找到系统输出设备。")
    if not device_arg:
        return sc.default_speaker()
    for spk in speakers:
        if device_arg.lower() in spk.name.lower():
            return spk
    sys.exit(f"找不到名称包含 '{device_arg}' 的设备。可用设备：\n"
             + "\n".join(f"  - {s.name}" for s in speakers))


def spectral_flux_envelope(signal, sr, frame_size=1024, hop=512):
    """计算简化的频谱通量（onset 包络）。"""
    n_frames = max(0, (len(signal) - frame_size) // hop + 1)
    if n_frames < 3:
        return np.array([], dtype=np.float64)

    window = np.hanning(frame_size).astype(np.float64)
    flux = np.empty(n_frames - 1, dtype=np.float64)
    prev_spectrum = None

    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size].astype(np.float64)
        if len(frame) < frame_size:
            break
        spectrum = np.abs(np.fft.rfft(frame * window))
        if prev_spectrum is not None:
            flux[i - 1] = np.sum(np.maximum(0.0, spectrum - prev_spectrum))
        prev_spectrum = spectrum

    return flux


def rms_onset_envelope(signal, frame_size=1024, hop=512):
    """能量差分的备选 onset 包络（速度更快，但不如频谱通量稳定）。"""
    n_frames = max(0, (len(signal) - frame_size) // hop + 1)
    if n_frames < 3:
        return np.array([], dtype=np.float64)

    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            break
        rms[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

    # 正向能量差分可以突出打击乐起始点
    diff = np.diff(rms)
    return np.maximum(0.0, diff)


def estimate_bpm(signal, sr, min_bpm=60.0, max_bpm=200.0, hop=512, mode="flux"):
    """从一段音频信号中估计 BPM。"""
    if mode == "rms":
        envelope = rms_onset_envelope(signal, hop=hop)
    else:
        envelope = spectral_flux_envelope(signal, sr, frame_size=hop * 2, hop=hop)

    if len(envelope) < 8:
        return None

    envelope = envelope - np.mean(envelope)
    if np.allclose(envelope, 0):
        return None

    # 自相关：找到节拍周期
    ac = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1:]
    if len(ac) < 2:
        return None

    # BPM -> lag（单位：帧）
    # lag = 60 * sr / (bpm * hop)
    min_lag = int(np.floor(60.0 * sr / (max_bpm * hop)))
    max_lag = int(np.ceil(60.0 * sr / (min_bpm * hop)))
    if min_lag < 1:
        min_lag = 1
    if max_lag >= len(ac):
        max_lag = len(ac) - 1
    if max_lag <= min_lag:
        return None

    # 用谐波加权选峰，减少半倍/双倍速误判
    best_lag = min_lag
    best_score = -np.inf
    for lag in range(min_lag, max_lag + 1):
        score = float(ac[lag])
        # 倍频程权重
        for mult in (2, 3):
            if lag * mult < len(ac):
                score += 0.5 * float(ac[lag * mult]) / mult
        if score > best_score:
            best_score = score
            best_lag = lag

    bpm = 60.0 * sr / (best_lag * hop)
    return bpm


class TempoTracker:
    """稍微平滑 BPM，并在半速/双速之间自动对齐。"""

    def __init__(self, history_size=10):
        self.history = deque(maxlen=history_size)

    def update(self, bpm):
        if self.history:
            ref = float(np.median(self.history))
            # 把新值拉到与历史中位数同一个倍频程附近
            while bpm < ref * 0.75:
                bpm *= 2.0
            while bpm > ref * 1.5:
                bpm /= 2.0
        self.history.append(bpm)
        return float(np.median(self.history))


def main():
    parser = argparse.ArgumentParser(description="实时检测系统正在播放音乐的 BPM")
    parser.add_argument("--list-devices", action="store_true",
                        help="列出可用输出设备后退出")
    parser.add_argument("--device", default=None,
                        help="输出设备名称关键词，例如 'Speakers' 或 '扬声器'")
    parser.add_argument("--samplerate", type=int, default=22050,
                        help="采样率，默认 22050")
    parser.add_argument("--window", type=float, default=8.0,
                        help="分析滑窗秒数，默认 8 秒")
    parser.add_argument("--update", type=float, default=1.0,
                        help="每隔多少秒更新一次 BPM，默认 1 秒")
    parser.add_argument("--chunk", type=int, default=2048,
                        help="每次读取的音频帧数")
    parser.add_argument("--min-bpm", type=float, default=60.0)
    parser.add_argument("--max-bpm", type=float, default=400.0)
    parser.add_argument("--mode", choices=["flux", "rms"], default="flux",
                        help="onset 检测方式：频谱通量(默认)或能量差分")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    speaker = select_speaker(args.device)
    sr = args.samplerate
    window_samples = int(sr * args.window)
    ring = np.zeros(window_samples, dtype=np.float32)
    tracker = TempoTracker()
    last_calc = 0.0

    print(f"正在从输出设备捕获：{speaker.name}")
    print(f"采样率 {sr} Hz，滑窗 {args.window:.1f}s，每 {args.update:.1f}s 更新一次")
    print("按 Ctrl+C 停止。")

    try:
        # soundcard 在 Windows 上默认就是 loopback，无需额外虚拟声卡；
        # macOS 可能需要 BlackHole，Linux 使用 PulseAudio/PipeWire monitor。
        with speaker.recorder(samplerate=sr, channels=1) as recorder:
            while True:
                data = recorder.record(numframes=args.chunk)
                if data is None or len(data) == 0:
                    continue

                mono = np.asarray(data, dtype=np.float32)
                if mono.ndim > 1:
                    mono = mono[:, 0]

                # 写入环形缓冲
                n = len(mono)
                if n >= len(ring):
                    ring[:] = mono[-len(ring):]
                else:
                    ring[:-n] = ring[n:]
                    ring[-n:] = mono

                now = time.monotonic()
                if now - last_calc >= args.update:
                    last_calc = now
                    bpm = estimate_bpm(ring, sr, args.min_bpm, args.max_bpm,
                                       hop=512, mode=args.mode)
                    if bpm is not None and 20.0 < bpm < 300.0:
                        smoothed = tracker.update(bpm)
                        print(f"\rBPM: {smoothed:6.1f}  (raw {bpm:6.1f})   ",
                              end="", flush=True)
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception as exc:
        print(f"\n捕获失败：{exc}", file=sys.stderr)
        print("提示：Windows 一般无需额外配置；macOS 请安装 BlackHole；",
              file=sys.stderr)
        print("Linux 请确认 PulseAudio/PipeWire 显示器源可用。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
