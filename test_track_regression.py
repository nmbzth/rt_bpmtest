# -*- coding: utf-8 -*-
"""test_track_regression.py — 已知曲目回归测试（流式节奏，模拟 app 0.5s 更新）。

覆盖 v2.1 修复：夢世界フォークロア 148 BPM 曾被节拍连续性正反馈锁死为 198。
用法: python test_track_regression.py
"""

import os

import numpy as np
import soundfile as sf

from bpm_strategies import STRATEGIES

SR = 22050
FFT = 1024
AUDIO = {"sample_rate": SR, "fft_size": FFT, "hop_size": 512}

# (文件, 标称 BPM, 容差, BPM范围或 None)
# 夢世界フォークロア：不加范围——专门防「连续性正反馈锁死 198」回归
TRACKS = [
    ("上海アリス幻樂団,黄昏フロンティア - 夢世界フォークロア.mp3", 148.0, 3.0, None),
    ("Supa7onyz - DESTRUCTION 3,2,1.flac", 321.321, 3.0, (300, 400)),
    ("MUSIC/rock_hiking_180.mp3", 180.0, 3.0, (168, 192)),
    ("MUSIC/electronic_rise_132.mp3", 132.0, 3.0, (120, 144)),
]


def load_mono(path, seconds=60.0):
    info = sf.info(path)
    sr2 = info.samplerate
    n = int(sr2 * min(seconds, info.duration))
    data, _ = sf.read(path, dtype="float32", always_2d=True, start=0, stop=n)
    mono = data.mean(axis=1)
    if sr2 != SR:
        old = np.linspace(0, 1, len(mono))
        new = np.linspace(0, 1, int(len(mono) * SR / sr2))
        mono = np.interp(new, old, mono).astype(np.float32)
    return mono


def stream_final(cls, mono, bpm_range=None):
    """模拟 app：逐块推入，每 0.5s 音频更新一次，返回最后一个 BPM。"""
    settings = {}
    if bpm_range:
        settings["min_bpm"], settings["max_bpm"] = bpm_range
    s = cls(settings=settings, audio_settings=AUDIO)
    last = -0.5
    final = None
    for i in range(0, len(mono), 2048):
        s.push_audio(mono[i:i + 2048])
        t = i / SR
        if t - last >= 0.5:
            last = t
            r = s.update()
            if r.bpm is not None:
                final = r.bpm
    return final


def main():
    failures = 0
    for path, true_bpm, tol, bpm_range in TRACKS:
        if not os.path.exists(path):
            print(f"[SKIP] {path} 不存在")
            continue
        mono = load_mono(path)
        for cls in STRATEGIES:
            final = stream_final(cls, mono, bpm_range)
            err = abs(final - true_bpm) if final else float("inf")
            ok = err <= tol
            if not ok:
                failures += 1
            print(f"  {'OK ' if ok else 'FAIL'} {cls.display_name:24s} "
                  f"{os.path.basename(path)[:32]:34s} 真实{true_bpm:.0f} 检测{final:.1f} 误差{err:.1f}")
    print("=" * 60)
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
