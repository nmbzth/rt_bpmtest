@echo off
REM 安装打包所需依赖并打包 Tkinter 版本实时 BPM 检测器
pip install soundcard numpy pyinstaller
pyinstaller --noconfirm --clean --windowed --onefile ^
  --name RTBPM ^
  --hidden-import soundcard ^
  --hidden-import numpy ^
  rt_bpm_app.py
