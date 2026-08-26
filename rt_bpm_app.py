# -*- coding: utf-8 -*-
"""带 Tkinter UI 的实时 BPM 检测软件。"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from bpm_strategies import BpmResult, STRATEGIES, STRATEGY_MAP


def list_sound_devices():
    """返回可用的系统输出设备列表。"""
    try:
        import soundcard as sc
    except ImportError:
        raise RuntimeError("未安装 soundcard，请先执行: pip install soundcard")
    speakers = sc.all_speakers()
    return [(i, spk.name) for i, spk in enumerate(speakers)]


class AudioCaptureThread(threading.Thread):
    """后台线程：读取系统 loopback 音频并调用当前策略分析。"""

    def __init__(self, device_name, sample_rate, fft_size, hop_size,
                 strategy, result_queue, stop_event):
        super().__init__(daemon=True)
        self.device_name = device_name
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.strategy = strategy
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.chunk_count = 0

    def run(self):
        try:
            import soundcard as sc
        except ImportError as exc:
            self.result_queue.put({"type": "error", "message": str(exc)})
            return

        try:
            speaker = None
            if self.device_name:
                for spk in sc.all_speakers():
                    if spk.name == self.device_name:
                        speaker = spk
                        break
            if speaker is None:
                speaker = sc.default_speaker()

            # soundcard 的 Speaker 不能直接录音，需要获取对应的 loopback Microphone。
            loopback_mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
            with loopback_mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while not self.stop_event.is_set():
                    data = recorder.record(numframes=2048)
                    if data is None or len(data) == 0:
                        continue
                    mono = np.asarray(data, dtype=np.float32)
                    if mono.ndim > 1:
                        mono = mono[:, 0]
                    # 每 3 个块发一次输入电平，用于 UI 电平条。
                    self.chunk_count += 1
                    if self.chunk_count % 3 == 0:
                        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
                        level = float(np.clip(rms * 10.0, 0.0, 1.0))
                        self.result_queue.put({"type": "level", "level": level})
                    result = self.strategy.process_audio(mono, self.sample_rate)
                    # v2.0：只有真正更新时才推送结果，不再每块重复刷 UI。
                    if result is not None:
                        self.result_queue.put(result)
        except Exception as exc:
            self.result_queue.put({"type": "error", "message": str(exc)})


class ScrollableFrame(ttk.Frame):
    """可滚动的设置容器，小窗口下也能完整操作所有控件。"""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def destroy(self):
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass
        super().destroy()


class SettingsPanel:
    """根据策略的 settings_schema 动态生成 Tkinter 设置界面。"""

    def __init__(self, parent, strategy_class):
        self.parent = parent
        self.strategy_class = strategy_class
        self.vars = {}
        self.entry_vars = {}
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=10, pady=5)
        self._build()

    def _build(self):
        title = ttk.Label(self.frame, text=self.strategy_class.display_name,
                          font=("Microsoft YaHei UI", 11, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        desc = ttk.Label(self.frame, text=self.strategy_class.description,
                         foreground="#555555", wraplength=480)
        desc.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        row = 2
        for field in self.strategy_class.settings_schema:
            key = field["key"]
            label = field["label"]
            ftype = field.get("type", "scale")

            label_widget = ttk.Label(self.frame, text=label)
            label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)

            if ftype == "scale":
                minv = float(field.get("min", 0))
                maxv = float(field.get("max", 100))
                resolution = float(field.get("resolution", 1))
                default = float(field.get("default", minv))
                var = tk.DoubleVar(value=default)
                self.vars[key] = var

                def fmt_val(v):
                    if resolution >= 1:
                        return f"{int(round(float(v)))}"
                    return f"{float(v):g}"

                entry_var = tk.StringVar(value=fmt_val(default))
                self.entry_vars[key] = entry_var

                scale = tk.Scale(
                    self.frame,
                    from_=minv,
                    to=maxv,
                    resolution=resolution,
                    orient="horizontal",
                    variable=var,
                    length=200,
                    showvalue=False,
                    command=lambda v, e=entry_var: e.set(fmt_val(v)),
                )
                scale.grid(row=row, column=1, sticky="w", pady=3)

                entry = ttk.Entry(self.frame, textvariable=entry_var, width=6)
                entry.grid(row=row, column=2, sticky="w", padx=(6, 2), pady=3)

                def on_entry(*_, var=var, entry_var=entry_var, minv=minv, maxv=maxv):
                    try:
                        val = float(entry_var.get())
                    except (ValueError, tk.TclError):
                        return
                    if val < minv:
                        val = minv
                    elif val > maxv:
                        val = maxv
                    var.set(val)

                entry_var.trace_add("write", on_entry)
            elif ftype == "combo":
                var = tk.StringVar(value=field.get("default", ""))
                self.vars[key] = var
                combo = ttk.Combobox(
                    self.frame,
                    textvariable=var,
                    values=list(field.get("values", [])),
                    state="readonly",
                    width=24,
                )
                combo.grid(row=row, column=1, sticky="w", pady=3)
            elif ftype == "check":
                var = tk.BooleanVar(value=bool(field.get("default", False)))
                self.vars[key] = var
                check = ttk.Checkbutton(self.frame, text="启用", variable=var)
                check.grid(row=row, column=1, sticky="w", pady=3)

            row += 1

        # 方案 7 的风格预设联动
        if hasattr(self.strategy_class, "PRESET_RANGES") and "preset" in self.vars:
            self._bind_preset()

    def _bind_preset(self):
        def on_preset(*_):
            name = self.vars["preset"].get()
            ranges = self.strategy_class.PRESET_RANGES
            if name in ranges and ranges[name] is not None:
                lo, hi = ranges[name]
                if "min_bpm" in self.vars:
                    self.vars["min_bpm"].set(float(lo))
                    self._sync_entry("min_bpm")
                if "max_bpm" in self.vars:
                    self.vars["max_bpm"].set(float(hi))
                    self._sync_entry("max_bpm")
        self.vars["preset"].trace_add("write", on_preset)

    def _sync_entry(self, key):
        if key in self.entry_vars and key in self.vars:
            self.entry_vars[key].set(f"{float(self.vars[key].get()):g}")

    def get_settings(self):
        result = {}
        for key, var in self.vars.items():
            if isinstance(var, tk.BooleanVar):
                result[key] = bool(var.get())
            elif isinstance(var, tk.DoubleVar):
                result[key] = float(var.get())
            else:
                result[key] = var.get()
        return result

    def destroy(self):
        self.frame.destroy()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("实时 BPM 检测器 v3.0")
        self.root.geometry("780x640")
        self.root.minsize(700, 580)

        self.result_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.settings_panel: SettingsPanel | None = None
        self.strategy = None

        self.device_var = tk.StringVar()
        self.sample_rate_var = tk.StringVar(value="22050")
        self.fft_var = tk.StringVar(value="1024")
        self.hop_var = tk.StringVar(value="128")
        self.mode_var = tk.StringVar()
        self.level_var = tk.DoubleVar(value=0.0)
        self.last_heartbeat = 0.0
        self.mode_var.set(STRATEGIES[0].display_name)

        self._build_top_bar()
        self._build_mode_bar()
        self._build_settings_area()
        self._build_display_area()

        self._load_devices()
        self._on_mode_change()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_queue)

    def _build_top_bar(self):
        top = ttk.LabelFrame(self.root, text="音频采集")
        top.pack(fill="x", padx=10, pady=(10, 5))

        # 第一行：设备与启停
        ttk.Label(top, text="输出设备:").grid(row=0, column=0, padx=(8, 4), pady=4, sticky="w")
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var,
                                         state="readonly", width=32)
        self.device_combo.grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(top, text="刷新", command=self._load_devices).grid(
            row=0, column=2, padx=4, pady=4)
        ttk.Button(top, text="说明", command=self._show_help).grid(
            row=0, column=3, padx=4, pady=4)
        self.start_btn = ttk.Button(top, text="开始监听", command=self._toggle_start_stop)
        self.start_btn.grid(row=0, column=4, padx=(12, 8), pady=4)

        # 第二行：音频精度参数，避免小窗口横向挤压
        ttk.Label(top, text="采样率:").grid(row=1, column=0, padx=(8, 4), pady=4, sticky="w")
        self.sr_combo = ttk.Combobox(top, textvariable=self.sample_rate_var,
                                     values=["16000", "22050", "32000", "44100", "48000"],
                                     state="readonly", width=8)
        self.sr_combo.grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="FFT:").grid(row=1, column=2, padx=(16, 4), pady=4, sticky="w")
        self.fft_combo = ttk.Combobox(top, textvariable=self.fft_var,
                                      values=["512", "1024", "2048"],
                                      state="readonly", width=7)
        self.fft_combo.grid(row=1, column=3, padx=4, pady=4, sticky="w")

        ttk.Label(top, text="Hop:").grid(row=1, column=4, padx=(16, 4), pady=4, sticky="w")
        self.hop_combo = ttk.Combobox(top, textvariable=self.hop_var,
                                      values=["64", "128", "256", "512"],
                                      state="readonly", width=7)
        self.hop_combo.grid(row=1, column=5, padx=4, pady=4, sticky="w")

    def _build_mode_bar(self):
        mode_frame = ttk.LabelFrame(self.root, text="检测方案")
        mode_frame.pack(fill="x", padx=10, pady=5)

        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=[cls.display_name for cls in STRATEGIES],
            state="readonly",
            width=30,
        )
        self.mode_combo.pack(side="left", padx=(8, 8), pady=6)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.mode_description = ttk.Label(mode_frame, text="", foreground="#555555")
        self.mode_description.pack(side="left", padx=4, pady=6)

    def _build_settings_area(self):
        self.settings_container = ScrollableFrame(self.root)
        self.settings_container.pack(fill="both", expand=True, padx=10, pady=5)

    def _build_display_area(self):
        display = ttk.LabelFrame(self.root, text="实时结果")
        display.pack(fill="x", padx=10, pady=(5, 10))

        self.bpm_label = ttk.Label(display, text="--", font=("Microsoft YaHei UI", 44, "bold"))
        self.bpm_label.pack(pady=(6, 0))

        self.status_label = ttk.Label(display, text="等待开始", font=("Microsoft YaHei UI", 11))
        self.status_label.pack()

        self.detail_label = ttk.Label(display, text="", foreground="#555555")
        self.detail_label.pack()

        self.extra_label = ttk.Label(display, text="", foreground="#333333")
        self.extra_label.pack()

        level_frame = ttk.Frame(display)
        level_frame.pack(pady=(4, 0))
        ttk.Label(level_frame, text="输入电平:").pack(side="left", padx=(0, 6))
        self.level_bar = ttk.Progressbar(
            level_frame,
            variable=self.level_var,
            maximum=1.0,
            length=260,
            mode="determinate",
        )
        self.level_bar.pack(side="left")

        self.candidates_text = tk.Text(display, height=3, width=60, state="disabled")
        self.candidates_text.pack(padx=8, pady=4)

    def _show_help(self):
        messagebox.showinfo(
            "FFT 与 Hop 说明",
            """FFT 窗口大小：
分析音频时每次取多长一段信号做频谱分析。
FFT 越大，频率分辨率越高，对低频和复杂音乐的识别更细致，但计算量也更大。

Hop 大小：
相邻两次 FFT 分析之间的前进步长。
Hop 越小，时间分辨率越高，BPM 更新越细腻，但计算频率更高；
Hop 越大，速度更快，但可能损失时间精度。

建议：日常使用 128/256；追求精度用 64；追求省电用 512。""",
        )

    def _load_devices(self):
        try:
            devices = list_sound_devices()
        except Exception as exc:
            self.device_combo["values"] = []
            self.device_var.set("")
            messagebox.showwarning("提示", f"无法获取音频设备：\n{exc}")
            return
        names = [name for _, name in devices]
        self.device_combo["values"] = names
        if names:
            self.device_var.set(names[0])

    def _on_mode_change(self, event=None):
        if self.worker is not None and self.worker.is_alive():
            self._stop()
        selected = self.mode_var.get()
        if not selected:
            return
        strategy_cls = None
        for cls in STRATEGIES:
            if cls.display_name == selected:
                strategy_cls = cls
                break
        if strategy_cls is None:
            return

        if self.settings_panel is not None:
            self.settings_panel.destroy()
            self.settings_panel = None

        self.settings_panel = SettingsPanel(self.settings_container.inner, strategy_cls)
        self.mode_description.config(text=strategy_cls.description)

    def _toggle_start_stop(self):
        if self.worker is not None and self.worker.is_alive():
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.device_var.get():
            messagebox.showwarning("提示", "请先选择输出设备。")
            return
        if self.settings_panel is None:
            messagebox.showwarning("提示", "请先选择检测方案。")
            return

        selected = self.mode_var.get()
        strategy_cls = None
        for cls in STRATEGIES:
            if cls.display_name == selected:
                strategy_cls = cls
                break
        if strategy_cls is None:
            return

        settings = self.settings_panel.get_settings()
        audio_settings = {
            "sample_rate": int(self.sample_rate_var.get()),
            "fft_size": int(self.fft_var.get()),
            "hop_size": int(self.hop_var.get()),
        }
        self.strategy = strategy_cls(settings=settings, audio_settings=audio_settings)

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Exception:
                break

        # 重新开始时立即清空上一次的数值和显示，避免旧数据污染。
        self._clear_display()
        self.level_var.set(0.0)
        self.last_heartbeat = time.monotonic()

        # 每个采集线程使用独立停止事件，避免旧线程被新启动误唤醒。
        self.stop_event = threading.Event()
        self.worker = AudioCaptureThread(
            device_name=self.device_var.get(),
            sample_rate=int(self.sample_rate_var.get()),
            fft_size=int(self.fft_var.get()),
            hop_size=int(self.hop_var.get()),
            strategy=self.strategy,
            result_queue=self.result_queue,
            stop_event=self.stop_event,
        )
        self.worker.start()
        self.start_btn.config(text="停止监听")
        self.status_label.config(text=f"正在监听：{self.device_var.get()}")

    def _stop(self):
        # 只发停止信号，不在 UI 线程 join，避免界面卡顿。
        self.stop_event.set()
        self.worker = None
        self.start_btn.config(text="开始监听")
        self.status_label.config(text="已停止")

    def _poll_queue(self):
        try:
            while True:
                item = self.result_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "error":
                    messagebox.showerror("错误", item.get("message", "未知错误"))
                    self._stop()
                    continue
                if isinstance(item, dict) and item.get("type") == "level":
                    self.level_var.set(float(item.get("level", 0.0)))
                    self.last_heartbeat = time.monotonic()
                    continue
                if isinstance(item, BpmResult):
                    self._update_result(item)
                    self.last_heartbeat = time.monotonic()
        except queue.Empty:
            pass

        # 防卡死看门狗：运行中如果长时间收不到电平/结果，自动停止。
        if self.worker is not None and self.worker.is_alive():
            if time.monotonic() - self.last_heartbeat > 8.0:
                messagebox.showwarning(
                    "提示",
                    """音频采集长时间没有数据，可能设备无输出或采集卡住。
已自动停止。""",
                )
                self._stop()

        self.root.after(100, self._poll_queue)

    def _clear_display(self):
        self.bpm_label.config(text="--")
        self.status_label.config(text="正在启动...")
        self.detail_label.config(text="")
        self.extra_label.config(text="")
        self.candidates_text.config(state="normal")
        self.candidates_text.delete("1.0", "end")
        self.candidates_text.config(state="disabled")

    def _update_result(self, result: BpmResult):
        if result is None:
            return

        if result.bpm is not None:
            self.bpm_label.config(text=f"{result.bpm:.3f}")
        else:
            self.bpm_label.config(text="--")

        conf_text = f"置信度 {result.confidence:.2f}" if result.confidence else ""
        self.status_label.config(text=f"{result.status}   {conf_text}")

        detail_parts = []
        if result.detail:
            detail_parts.append(result.detail)
        if result.window_seconds:
            detail_parts.append(f"窗口 {result.window_seconds:.1f}s")
        if getattr(result, "proc_ms", 0.0) > 0:
            detail_parts.append(f"更新 {result.proc_ms:.1f}ms")
        self.detail_label.config(text="  |  ".join(detail_parts))

        extra = []
        if getattr(result, "beats", 0) > 0:
            sigma = getattr(result, "sigma_bpm", 0.0)
            if sigma > 0:
                extra.append(f"±{sigma:.4f} ({result.beats}拍)")
            else:
                extra.append(f"{result.beats}拍")
        if result.fast_bpm is not None:
            extra.append(f"快速 {result.fast_bpm:.3f}")
        if result.stable_bpm is not None:
            extra.append(f"稳定 {result.stable_bpm:.3f}")
        if result.raw_bpm is not None and result.raw_bpm != result.bpm:
            extra.append(f"原始 {result.raw_bpm:.3f}")
        self.extra_label.config(text="   ".join(extra))

        if result.candidates:
            lines = [f"{bpm:.1f} BPM  score {score:.2f}" for bpm, score in result.candidates]
            self.candidates_text.config(state="normal")
            self.candidates_text.delete("1.0", "end")
            self.candidates_text.insert("1.0", "\n".join(lines))
            self.candidates_text.config(state="disabled")
        else:
            self.candidates_text.config(state="normal")
            self.candidates_text.delete("1.0", "end")
            self.candidates_text.config(state="disabled")

    def on_close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names() or "winnative" in style.theme_names():
            style.theme_use("vista" if "vista" in style.theme_names() else "winnative")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
