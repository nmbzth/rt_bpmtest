# -*- coding: utf-8 -*-
"""bench_compare.py — v2.0 性能与精度验证脚本。

用法:
    python bench_compare.py

输出:
  [1] 合成节拍器精度（60-200 每 5 BPM + 极端 30-60/200-400）
  [2] 真实音乐 A/B：flux / wflux / rms 三种 onset 方法（±12 BPM 范围）
  [3] 性能：单次 update 耗时、每块 push 耗时
  [4] 30s 流式：总 CPU、结果条数、首个 BPM 延迟
"""

import time
import os

import numpy as np
import soundfile as sf

from bpm_strategies import STRATEGIES, STRATEGY_MAP

SR = 22050
AUDIO = {"sample_rate": SR, "fft_size": 1024, "hop_size": 512}

TRACKS = [
    ("rock_hiking_180.mp3", 180.0),
    ("classical_mountain_king_160.mp3", 160.0),
    ("classical_sugar_plum_120.mp3", 120.0),
    ("funk_autumn_dance_106.mp3", 106.0),
    ("jazz_late_night_67.mp3", 67.0),
    ("jazz_summer_104.mp3", 104.0),
    ("electronic_rise_132.mp3", 132.0),
    ("electronic_golden_95.mp3", 95.0),
]


def make_click(bpm, dur, sr=SR):
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    k = 0
    while True:
        i = int(round(k * period * sr))
        if i >= n:
            break
        length = max(8, int(0.05 * sr))
        if i + length < n:
            x[i:i + length] += np.exp(-np.linspace(0, 8, length)).astype(np.float32)
        k += 1
    return x


def load_mono(name, seconds=30.0):
    path = os.path.join("MUSIC", name)
    info = sf.info(path)
    sr2 = info.samplerate
    n = int(sr2 * min(seconds, info.duration))
    data, _ = sf.read(path, dtype="float32", always_2d=True, start=0, stop=n)
    mono = data.mean(axis=1)
    if sr2 != 22050:
        old = np.linspace(0, 1, len(mono))
        new = np.linspace(0, 1, int(len(mono) * 22050 / sr2))
        mono = np.interp(new, old, mono).astype(np.float32)
    return mono


def metronome_err(cls, bpm, dur, min_bpm, max_bpm):
    x = make_click(bpm, dur)
    s = cls(settings={"min_bpm": min_bpm, "max_bpm": max_bpm}, audio_settings=AUDIO)
    for i in range(0, len(x), 2048):
        s.push_audio(x[i:i + 2048])
    res = s.update()
    est = res.bpm if res.bpm is not None else float("nan")
    return abs(est - bpm) if est == est else float("nan")


def music_ranged(cls_key, method, name, true_bpm):
    mono = load_mono(name)
    settings = {"min_bpm": max(40, true_bpm - 12), "max_bpm": min(300, true_bpm + 12),
                "method": method}
    s = STRATEGY_MAP[cls_key](settings=settings, audio_settings=AUDIO)
    for i in range(0, len(mono), 2048):
        s.push_audio(mono[i:i + 2048])
    res = s.update()
    est = res.bpm if res.bpm is not None else float("nan")
    return est, abs(est - true_bpm) if est == est else float("nan")


def main():
    print("=" * 78)
    print("[1] 合成节拍器最大误差")
    bpms = list(range(60, 205, 5))
    print(f"    60-200 每5BPM (14s):")
    for cls in STRATEGIES:
        errs = [metronome_err(cls, b, 14.0, 60.0, 200.0) for b in bpms]
        errs = [e for e in errs if e == e]
        print(f"      {cls.display_name}: {max(errs):.3f}")
    bpms_x = list(range(30, 65, 5)) + list(range(200, 405, 5))
    print(f"    30-60 / 200-400 每5BPM (16s):")
    for cls in STRATEGIES:
        errs = [metronome_err(cls, b, 16.0, 25.0, 410.0) for b in bpms_x]
        errs = [e for e in errs if e == e]
        print(f"      {cls.display_name}: {max(errs):.3f}")

    print("=" * 78)
    print("[2] 真实音乐 A/B：onset 方法（策略2 两级快速/稳定, ±12 BPM 范围）")
    for method in ("flux", "wflux", "rms"):
        row = []
        for name, true_bpm in TRACKS:
            est, err = music_ranged("fast_stable", method, name, true_bpm)
            row.append(f"{true_bpm:.0f}->{est:.1f}({err:.1f})")
        print(f"    {method:6s}: " + "  ".join(row))
    print("    注：古典 mountain_king 为已知节拍脉冲歧义曲目")

    print("=" * 78)
    print("[3] 性能：单次 update 耗时（15s 120BPM 节拍器，满缓冲）")
    x = make_click(120, 15.0)
    for cls in STRATEGIES:
        s = cls(settings={}, audio_settings=AUDIO)
        for i in range(0, len(x), 2048):
            s.push_audio(x[i:i + 2048])
        s.update()
        t0 = time.perf_counter()
        for _ in range(10):
            s.update()
        print(f"    {cls.display_name}: {(time.perf_counter() - t0) / 10 * 1000:.2f} ms")

    print("=" * 78)
    print("[4] 30s 真实音乐流式（rock_hiking_180, 2048 帧块）")
    mono = load_mono("rock_hiking_180.mp3")
    for cls in STRATEGIES:
        s = cls(settings={}, audio_settings=AUDIO)
        cpu = 0.0
        n_res = 0
        need = None
        for i in range(0, len(mono), 2048):
            t0 = time.perf_counter()
            r = s.process_audio(mono[i:i + 2048], 22050)
            cpu += time.perf_counter() - t0
            if r is not None:
                n_res += 1
                if r.bpm is not None and need is None:
                    need = i + 2048
        print(f"    {cls.display_name}: CPU {cpu*1000:.1f} ms, 结果 {n_res} 条, "
              f"首个 BPM {need/SR:.2f}s" if need else
              f"    {cls.display_name}: CPU {cpu*1000:.1f} ms, 结果 {n_res} 条")


if __name__ == "__main__":
    main()
