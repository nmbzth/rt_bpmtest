# -*- coding: utf-8 -*-
"""用 MUSIC 下的真实音乐离线测试 7 种策略。"""
import os
import numpy as np
import soundfile as sf
from bpm_strategies import STRATEGIES

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

def load_mono(path, seconds=30.0):
    info = sf.info(path)
    sr = info.samplerate
    n = int(sr * min(seconds, info.duration))
    data, sr2 = sf.read(path, dtype='float32', always_2d=True, start=0, stop=n)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    # 降到 22050 方便测试
    if sr2 != 22050:
        # 简单重采样：线性插值
        target_len = int(len(mono) * 22050.0 / sr2)
        old_x = np.linspace(0, 1, len(mono), dtype=np.float64)
        new_x = np.linspace(0, 1, target_len, dtype=np.float64)
        mono = np.interp(new_x, old_x, mono).astype(np.float32)
    return mono, 22050

def main():
    print('Strategy\t' + '\t'.join(name.split('_')[0] for name, _ in TRACKS))
    for cls in STRATEGIES:
        errs = []
        vals = []
        for name, true_bpm in TRACKS:
            path = os.path.join('MUSIC', name)
            mono, sr = load_mono(path, seconds=30.0)
            s = cls(settings={}, audio_settings={'sample_rate': sr, 'fft_size': 1024, 'hop_size': 512})
            for i in range(0, len(mono), 2048):
                s.push_audio(mono[i:i+2048])
            res = s.update()
            est = res.bpm if res.bpm is not None else float('nan')
            vals.append(f'{est:.1f}')
            errs.append(abs(est - true_bpm) if est == est else float('nan'))
        print(f'{cls.display_name}\t' + '\t'.join(vals) + f'\tmaxerr={np.nanmax(errs):.2f}')

if __name__ == '__main__':
    main()
