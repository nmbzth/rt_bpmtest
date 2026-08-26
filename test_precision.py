# -*- coding: utf-8 -*-
"""v3.0 三位小数 BPM 精度验证：合成稳态节拍器。

判据（60s）：
  |估计 - 真值| <= 0.0015 BPM
  sigma_bpm           <= 0.0005 BPM
同时打印 15s / 30s / 60s 的 σ 收敛情况。
"""

import time

import numpy as np

from bpm_strategies import STRATEGIES

SR = 22050
FFT = 1024
HOP = 512
AUDIO = {"sample_rate": SR, "fft_size": FFT, "hop_size": HOP}

# 非整数/整数、快/中/慢 BPM 覆盖
BPMS = [321.321, 317.777, 320.000, 180.000, 148.000, 65.000]
DURATIONS = [15.0, 30.0, 60.0]
TOL = 0.0015
SIG_TOL = 0.0005


def make_click(bpm: float, dur: float = 60.0, sr: int = SR) -> np.ndarray:
    """精确相位合成节拍器：i = round(k*60/bpm*sr)，指数衰减真实包络。"""
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    k = 0
    while True:
        i = int(round(k * period * sr))
        if i >= n:
            break
        length = max(8, int(0.03 * sr))
        if i + length < n:
            env = np.exp(-np.linspace(0, 8, length)).astype(np.float32)
            x[i:i + length] += env
        k += 1
    return x


def run_one(cls, bpm: float, dur: float):
    x = make_click(bpm, dur)
    settings = {"min_bpm": 30.0, "max_bpm": 410.0}
    s = cls(settings=settings, audio_settings=AUDIO)
    for i in range(0, len(x), 2048):
        s.push_audio(x[i:i + 2048])
    t0 = time.perf_counter()
    res = s.update()
    res.proc_ms = (time.perf_counter() - t0) * 1000.0
    return res


def main():
    failures = 0
    print("=" * 100)
    print("v3.0 合成精度：FastStable 代表策略（所有策略共用 BaseStrategy 细化链）")
    print(f"{'BPM':>10} {'时长':>5} {'估计':>12} {'误差':>10} {'σ_BPM':>10} {'拍数':>5}  耗时ms")
    for bpm in BPMS:
        for dur in DURATIONS:
            res = run_one(STRATEGIES[1], bpm, dur)
            err = res.bpm - bpm if res.bpm is not None else float("nan")
            ok = abs(err) <= TOL and res.sigma_bpm <= SIG_TOL
            if not ok:
                failures += 1
            print(f"{bpm:10.3f} {dur:5.0f} {res.bpm:12.6f} {err:+10.6f} "
                  f"{res.sigma_bpm:10.6f} {res.beats:5d}  {res.proc_ms:7.2f} "
                  f"{'OK' if ok else 'FAIL'}")

    print("=" * 100)
    print("60s 全策略判据")
    for cls in STRATEGIES:
        for bpm in BPMS:
            res = run_one(cls, bpm, 60.0)
            err = res.bpm - bpm if res.bpm is not None else float("nan")
            ok = abs(err) <= TOL and res.sigma_bpm <= SIG_TOL
            if not ok:
                failures += 1
            print(f"  {cls.display_name:24s} {bpm:8.3f} -> {res.bpm:12.6f} "
                  f"err={err:+9.6f} σ={res.sigma_bpm:.6f} beats={res.beats:3d} "
                  f"{'OK' if ok else 'FAIL'}")

    print("=" * 100)
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
