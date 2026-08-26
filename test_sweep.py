# -*- coding: utf-8 -*-
"""60-200 每5BPM 的合成节拍器精度测试。"""
import numpy as np
from bpm_strategies import STRATEGIES

SR = 22050
DUR = 14.0
FFT = 1024
HOP = 512
AUDIO = {'sample_rate': SR, 'fft_size': FFT, 'hop_size': HOP}

def make_click(bpm, dur=DUR, sr=SR):
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
            env = np.exp(-np.linspace(0, 8, length)).astype(np.float32)
            x[i:i+length] += env
        k += 1
    return x

def test_strategy(cls, bpm):
    x = make_click(bpm)
    s = cls(settings={}, audio_settings=AUDIO)
    for i in range(0, len(x), 2048):
        s.push_audio(x[i:i+2048])
    res = s.update()
    return res

def main():
    bpms = list(range(60, 205, 5))
    print('BPM\t' + '\t'.join(c.display_name for c in STRATEGIES))
    for bpm in bpms:
        ests = []
        for cls in STRATEGIES:
            res = test_strategy(cls, bpm)
            est = res.bpm if res.bpm is not None else float('nan')
            ests.append(est)
        print(f'{bpm}\t' + '\t'.join(f'{e:.1f}' if e == e else '--' for e in ests))

if __name__ == '__main__':
    main()
