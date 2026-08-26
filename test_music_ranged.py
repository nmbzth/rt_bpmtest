# -*- coding: utf-8 -*-
"""用真实音乐 + 紧 BPM 范围测试。"""
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

def load_mono(name, seconds=30.0):
    path=os.path.join('MUSIC',name); info=sf.info(path); sr=info.samplerate; n=int(sr*min(seconds,info.duration))
    data,sr2=sf.read(path,dtype='float32',always_2d=True,start=0,stop=n)
    mono=data.mean(axis=1)
    if sr2!=22050:
        old=np.linspace(0,1,len(mono)); new=np.linspace(0,1,int(len(mono)*22050/sr2))
        mono=np.interp(new,old,mono).astype(np.float32)
    return mono

def main():
    print('Strategy\t' + '\t'.join(name.split('_')[0] for name,_ in TRACKS))
    for cls in STRATEGIES:
        vals=[]
        for name, true_bpm in TRACKS:
            mono=load_mono(name)
            settings={'min_bpm': max(40,true_bpm-12), 'max_bpm': min(300,true_bpm+12)}
            s=cls(settings=settings, audio_settings={'sample_rate':22050,'fft_size':1024,'hop_size':512})
            for i in range(0,len(mono),2048): s.push_audio(mono[i:i+2048])
            res=s.update(); est=res.bpm if res.bpm else float('nan')
            vals.append(f'{est:.1f}')
        print(f'{cls.display_name}\t'+'\t'.join(vals))

if __name__=='__main__':
    main()
