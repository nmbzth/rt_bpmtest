# -*- coding: utf-8 -*-
"""极端 BPM 扫描：30-60 与 200-400，每5BPM。"""
import numpy as np
from bpm_strategies import STRATEGIES

SR = 22050
DUR = 16.0
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
        length = max(8, int(0.03 * sr))
        if i + length < n:
            env = np.exp(-np.linspace(0, 8, length)).astype(np.float32)
            x[i:i+length] += env
        k += 1
    return x

def test(cls, bpm):
    x = make_click(bpm)
    settings = {'min_bpm': 25.0, 'max_bpm': 410.0}
    s = cls(settings=settings, audio_settings=AUDIO)
    for i in range(0, len(x), 2048):
        s.push_audio(x[i:i+2048])
    return s.update()

def main():
    bpms = list(range(30, 65, 5)) + list(range(200, 405, 5))
    maxerr = {c.display_name: 0.0 for c in STRATEGIES}
    print('BPM\t' + '\t'.join(c.display_name for c in STRATEGIES))
    for bpm in bpms:
        vals = []
        for cls in STRATEGIES:
            res = test(cls, bpm)
            est = res.bpm if res.bpm is not None else float('nan')
            vals.append(est)
            if est == est:
                maxerr[cls.display_name] = max(maxerr[cls.display_name], abs(est - bpm))
        print(f'{bpm}\t' + '\t'.join(f'{e:.2f}' if e == e else '--' for e in vals))
    print('\nMAX ERR')
    for k, v in maxerr.items():
        print(k, round(v, 4))

if __name__ == '__main__':
    main()
