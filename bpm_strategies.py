# -*- coding: utf-8 -*-
"""BPM 检测策略 v2.0：包含 7 种可选方案，以及通用的实时分析接口。

v2.0 相对 v1.x 的性能与精度改进：
1. OnsetEngine 增量式 onset 包络：每块音频只计算新到达的帧（FFT 批处理），
   不再每次更新都整窗重算（原先 8s 窗口约 27ms/次）。
2. 环形音频缓冲：push_audio 不再整段 memmove。
3. FFT 自相关 + 无偏归一化（除以 (N-lag)），去除短 lag 偏好。
4. 谐波加权选峰（comb scoring），半速/倍速歧义更稳健。
5. 节拍连续性：估计时参考上一稳定 BPM，加权倍频一致的候选。
6. 加权多频段 flux（wflux）：抑制 <60Hz 轰隆与 >10kHz 噪声，
   增强 100Hz-4kHz 主节奏频带。
7. process_audio 只在有新结果时返回，不再每块刷 UI。
8. BpmResult.proc_ms 记录单次更新耗时。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BUFFER_SECONDS = 60.0
PRECISION_SECONDS = 30.0
VERSION = "3.0"


@dataclass
class BpmResult:
    mode: str = ""
    bpm: Optional[float] = None
    raw_bpm: Optional[float] = None
    fast_bpm: Optional[float] = None
    stable_bpm: Optional[float] = None
    confidence: float = 0.0
    status: str = "等待音频"
    window_seconds: float = 0.0
    candidates: List[Tuple[float, float]] = field(default_factory=list)
    detail: str = ""
    proc_ms: float = 0.0
    sigma_bpm: float = 0.0
    beats: int = 0


# ---------------------------------------------------------------------------
# 通用音频分析函数（批处理版，保持 v1 兼容）
# ---------------------------------------------------------------------------

def spectral_flux_envelope(signal: np.ndarray, fft_size: int = 1024,
                           hop: int = 512) -> np.ndarray:
    """计算频谱通量 onset 包络（批量 FFT 版本）。"""
    return compute_onset_envelope(signal, fft_size, hop, method="flux")


def rms_onset_envelope(signal: np.ndarray, fft_size: int = 1024,
                       hop: int = 512) -> np.ndarray:
    """能量差分 onset 包络，速度更快。"""
    return compute_onset_envelope(signal, fft_size, hop, method="rms")


def _band_weight(fft_size: int, sr: int) -> np.ndarray:
    """多频段权重：<60Hz 滚降、100Hz-4kHz 增强、>10kHz 滚降。"""
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    w = np.minimum(1.0, freqs / 60.0)
    mid = (freqs >= 100.0) & (freqs <= 4000.0)
    w[mid] *= 1.5
    high = freqs > 10000.0
    w[high] *= np.maximum(0.0, 1.0 - (freqs[high] - 10000.0) / 6000.0)
    w[0] = 0.0
    return w


def compute_onset_envelope(signal: np.ndarray, fft_size: int = 1024,
                           hop: int = 512, method: str = "flux",
                           sr: Optional[int] = None) -> np.ndarray:
    """批量计算 onset 包络（flux / wflux / rms）。FFT 帧批量处理。"""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = max(0, (len(signal) - fft_size) // hop + 1)
    if n < 3:
        return np.array([], dtype=np.float64)

    idx = (hop * np.arange(n, dtype=np.int64)[:, None]
           + np.arange(fft_size, dtype=np.int64)[None, :])
    frames = signal[idx]
    window = np.hanning(fft_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float64)

    if method in ("flux", "wflux"):
        if method == "wflux" and sr is not None:
            spec = spec * _band_weight(fft_size, int(sr))
        diff = spec[1:] - spec[:-1]
        return np.sum(np.maximum(0.0, diff), axis=1)

    # rms
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return np.maximum(0.0, np.diff(rms))


def estimate_bpm_from_envelope(
    env: np.ndarray,
    sr: int,
    hop: int,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
    ref_bpm: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """无偏自相关 + 谐波加权选峰 + 抛物线插值的 BPM 估计。

    返回 (bpm, confidence)。
    """
    if env is None or len(env) < 8:
        return None, 0.0
    x = env - np.mean(env)
    if np.allclose(x, 0.0):
        return None, 0.0
    n = len(x)

    min_lag = max(1, int(np.ceil(60.0 * sr / (max_bpm * hop))))
    max_lag = min(n - 1, int(np.floor(60.0 * sr / (min_bpm * hop))))
    if max_lag <= min_lag:
        return None, 0.0

    # FFT 自相关（O(N log N)），再除以重叠长度得到无偏归一化
    fft_n = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, fft_n)
    ac = np.fft.irfft(spec * np.conj(spec), fft_n)[:n]
    ac = ac / np.maximum(1.0, np.arange(n, 0, -1, dtype=np.float64))
    ac0 = max(float(ac[0]), 1e-12)

    # 参考 lag：用于节拍连续性加权（倍频一致 → 加分）
    ref_k: Optional[float] = None
    if ref_bpm is not None and min_bpm <= ref_bpm <= max_bpm:
        ref_k = 60.0 * sr / (ref_bpm * hop)

    harmonic_weights = ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25), (6, 0.2))

    def sample_ac(m: float) -> float:
        """在连续位置 m 处线性插值采样 ac（m 为小数 lag）。"""
        if m < 0.0 or m >= n - 1:
            return 0.0
        i0 = int(np.floor(m))
        frac = m - i0
        return float((1.0 - frac) * ac[i0] + frac * ac[i0 + 1])

    def lag_score_raw(lag: float) -> float:
        s = 0.0
        for k, w in harmonic_weights:
            s += w * sample_ac(lag * k)
        return s / ac0

    def continuity_factor(lag: float) -> float:
        """参考 BPM 的倍频一致性增益。"""
        if ref_k is None:
            return 1.0
        octv = float(np.log2(lag / ref_k))
        # octv 接近整数 → 与参考倍频一致 → 加分
        return 1.0 + 0.6 * float(np.exp(-(octv - round(octv)) ** 2 / 0.05))

    # 候选：局部极大值 + 抛物线定位小数 lag
    candidates: List[Tuple[int, float, float, float]] = []  # (整数lag, 小数lag, 原始分, 连续性分)
    for lag in range(min_lag, max_lag + 1):
        left_ok = (lag == min_lag) or (float(ac[lag]) >= float(ac[lag - 1]))
        right_ok = (lag == max_lag) or (float(ac[lag]) >= float(ac[lag + 1]))
        if not (left_ok and right_ok):
            continue
        lf = float(lag)
        if 0 < lag < n - 1:
            y0, y1, y2 = float(ac[lag - 1]), float(ac[lag]), float(ac[lag + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                lf = lag + float(np.clip(delta, -0.5, 0.5))
        s_raw = lag_score_raw(lf)
        candidates.append((lag, lf, s_raw, s_raw * continuity_factor(lf)))
    if not candidates:
        lag0 = int(np.argmax(ac[min_lag:max_lag + 1])) + min_lag
        lf0 = float(lag0)
        s0 = lag_score_raw(lf0)
        candidates = [(lag0, lf0, s0, s0 * continuity_factor(lf0))]

    # 连续性只作 tie-breaker：原始得分低于最佳候选 90% 的候选不享受
    # 连续性加分——否则一旦某时刻误判（如曲目内存在 4/3 关系的强韵律层），
    # 错误候选会借 ×1.6 增益形成正反馈永久锁死，盖过真正的基频。
    best_raw = max(c[2] for c in candidates)
    finals = [(lag, lf, s_cont if s_raw >= best_raw * 0.9 else s_raw)
              for lag, lf, s_raw, s_cont in candidates]

    best_final = max(f[2] for f in finals)
    # 短周期优先：从短 lag 起取第一个达到 0.85×最佳得分的候选，
    # 避免整数倍频（半速/三分速）误判。
    best_lag, best_lag_f = finals[0][0], finals[0][1]
    for lag, lf, s in sorted(finals, key=lambda t: t[0]):
        if s >= best_final * 0.85:
            best_lag, best_lag_f = lag, lf
            break

    bpm = 60.0 * sr / (best_lag_f * hop)
    confidence = float(np.clip(ac[best_lag] / ac0, 0.0, 1.0))
    return bpm, confidence


def estimate_bpm_autocorr(
    signal: np.ndarray,
    sr: int,
    fft_size: int = 1024,
    hop: int = 512,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
    method: str = "flux",
    ref_bpm: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """兼容 v1 签名：从整段信号一次性估计 BPM。"""
    env = compute_onset_envelope(signal, fft_size, hop, method=method, sr=sr)
    return estimate_bpm_from_envelope(env, sr, hop, min_bpm, max_bpm, ref_bpm)


def find_onset_times_from_env(env: np.ndarray, sr: int, hop: int,
                              threshold: float = 0.2,
                              min_gap: float = 0.2) -> List[float]:
    """从 onset 包络中找峰值时间点（相对包络起点）。阈值是归一化相对阈值。"""
    if env is None or len(env) < 3:
        return []
    env = env - env.mean()
    peak = float(np.max(env))
    if peak <= 1e-9:
        return []

    env = env / peak
    thr = float(threshold)
    times: List[float] = []
    last_time = -min_gap * 2

    for i in range(1, len(env) - 1):
        if env[i] >= thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            y0, y1, y2 = float(env[i - 1]), float(env[i]), float(env[i + 1])
            denom = y0 - 2.0 * y1 + y2
            delta = 0.0
            if abs(denom) > 1e-12:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
            t = ((i + delta) * hop) / sr
            if t - last_time >= min_gap:
                times.append(t)
                last_time = t
    return times


def find_onset_times(signal: np.ndarray, sr: int, fft_size: int = 1024,
                     hop: int = 512, threshold: float = 0.2,
                     min_gap: float = 0.2) -> List[float]:
    """兼容 v1 签名：从整段信号一次性找 onset 时间。"""
    env = compute_onset_envelope(signal, fft_size, hop, method="flux")
    return find_onset_times_from_env(env, sr, hop, threshold, min_gap)


# ---------------------------------------------------------------------------
# OnsetEngine：增量式 onset 包络引擎
# ---------------------------------------------------------------------------

class OnsetEngine:
    """增量式 onset 包络引擎。

    内部维护音频环形缓冲 + 频谱通量包络环形缓冲。
    每次 push 只计算新到达的帧（FFT 批处理），计算量 O(新增样本)。
    """

    def __init__(self, sr: int, fft_size: int, hop: int, method: str = "flux",
                 buffer_seconds: float = BUFFER_SECONDS):
        self.sr = int(sr)
        self.fft_size = int(fft_size)
        self.hop = int(hop)
        self.method = str(method)
        self.window = np.hanning(self.fft_size).astype(np.float32)

        self.audio_cap = int(self.sr * buffer_seconds * 1.3) + self.fft_size
        self.audio = np.zeros(self.audio_cap, dtype=np.float32)
        self.audio_head = 0          # 环形写指针
        self.audio_total = 0         # 已接收总样本数

        self.env_cap = int(buffer_seconds * self.sr / self.hop) + 16
        self.env = np.zeros(self.env_cap, dtype=np.float64)
        self.env_head = 0            # 环形写指针
        self.env_count = 0           # 有效包络点数

        self.next_frame = 0          # 下一个待计算帧的起始样本（从 0 起）
        self.prev_spec: Optional[np.ndarray] = None
        self.prev_rms: Optional[float] = None

        self.band_weight: Optional[np.ndarray] = None
        if self.method == "wflux":
            self.band_weight = _band_weight(self.fft_size, self.sr).astype(np.float64)

    def reset(self) -> None:
        self.audio_total = 0
        self.audio_head = 0
        self.env_count = 0
        self.env_head = 0
        self.next_frame = 0
        self.prev_spec = None
        self.prev_rms = None

    def push(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        if n >= self.audio_cap:
            # 极端情况：一次性超大块，重置到块尾
            self.audio[:] = samples[-self.audio_cap:]
            self.audio_head = 0
            self.audio_total = self.audio_cap
            self.next_frame = 0
            self.env_count = 0
            self.env_head = 0
            self.prev_spec = None
            self.prev_rms = None
            self._compute_new_frames()
            return

        head = int(self.audio_head)
        if head + n <= self.audio_cap:
            self.audio[head:head + n] = samples
            self.audio_head = head + n
        else:
            k = self.audio_cap - head
            self.audio[head:] = samples[:k]
            self.audio[:n - k] = samples[k:]
            self.audio_head = n - k
        self.audio_total += n
        self._compute_new_frames()

    def _compute_new_frames(self) -> None:
        # 注意：total 用无上限的 audio_total（环形映射由 % audio_cap 完成），
        # 否则音频回绕后新帧停止计算，包络变成过期数据。
        total = self.audio_total
        fft = self.fft_size
        hop = self.hop
        if total < fft:
            return
        n_new = (total - fft - self.next_frame) // hop + 1
        if n_new <= 0:
            return
        starts = self.next_frame + hop * np.arange(n_new, dtype=np.int64)
        self.next_frame = int(starts[-1]) + hop

        idx = (np.arange(fft, dtype=np.int64)[None, :]
               + starts[:, None]) % self.audio_cap
        frames = self.audio[idx]  # (M, fft) float32
        spec = np.abs(np.fft.rfft(frames * self.window, axis=1)).astype(np.float64)

        if self.method in ("flux", "wflux"):
            if self.band_weight is not None:
                spec = spec * self.band_weight
            if self.prev_spec is None:
                self.prev_spec = spec[-1:]
                if spec.shape[0] > 1:
                    # 首块内部帧差（帧 1..M-1 vs 前一帧）
                    diff = spec[1:] - spec[:-1]
                    self._append_env(np.sum(np.maximum(0.0, diff), axis=1))
            else:
                diff = spec - np.vstack((self.prev_spec, spec[:-1]))
                self._append_env(np.sum(np.maximum(0.0, diff), axis=1))
                self.prev_spec = spec[-1:]
        else:  # rms
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            if self.prev_rms is None:
                self.prev_rms = float(rms[-1])
                if rms.shape[0] > 1:
                    d = rms[1:] - rms[:-1]
                    self._append_env(np.maximum(0.0, d))
            else:
                diff = rms - np.concatenate(([self.prev_rms], rms[:-1]))
                self._append_env(np.maximum(0.0, diff))
                self.prev_rms = float(rms[-1])

    def _append_env(self, values: np.ndarray) -> None:
        m = len(values)
        if m == 0:
            return
        if m >= self.env_cap:
            values = values[-self.env_cap:]
            m = self.env_cap
        head = int(self.env_head)
        if head + m <= self.env_cap:
            self.env[head:head + m] = values
        else:
            k = self.env_cap - head
            self.env[head:] = values[:k]
            self.env[:m - k] = values[k:]
        self.env_head = (head + m) % self.env_cap
        self.env_count = min(self.env_cap, self.env_count + m)

    def env_tail(self, seconds: Optional[float] = None) -> np.ndarray:
        """最近的 k 个包络值（按时间先后排序）。"""
        if self.env_count == 0:
            return np.zeros(0, dtype=np.float64)
        if seconds is None:
            k = self.env_count
        else:
            k = min(self.env_count, max(1, int(seconds * self.sr / self.hop)))
        if k >= self.env_cap:
            return np.concatenate((self.env[self.env_head:], self.env[:self.env_head]))
        head = int(self.env_head)
        start = head - k
        if start >= 0:
            return self.env[start:head].copy()
        left = k - head
        return np.concatenate((self.env[self.env_cap - left:], self.env[:head]))

    def env_tail_with_start(self, seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """返回最近包络及该包络第一个帧的绝对起始样本序号。"""
        env = self.env_tail(seconds)
        k = len(env)
        first_start = self.next_frame - self.hop * k
        return env, int(first_start)

    def audio_tail(self, seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """返回最近原始音频（按时间先后）及其第一个样本的绝对序号。"""
        if seconds is None:
            n = min(self.audio_total, self.audio_cap)
        else:
            n = min(self.audio_total, self.audio_cap, int(seconds * self.sr))
        if n <= 0:
            return np.zeros(0, dtype=np.float32), 0
        start = self.audio_total - n
        idx = (start + np.arange(n, dtype=np.int64)) % self.audio_cap
        return self.audio[idx].copy(), int(start)


# ---------------------------------------------------------------------------
# v3.0：子采样 onset 定位 + 加权最小二乘周期回归
# ---------------------------------------------------------------------------

def refine_onset_times_from_audio(
    env: np.ndarray,
    first_start: int,
    audio: np.ndarray,
    audio_start: int,
    sr: int,
    hop: int,
    coarse_bpm: float,
    threshold: float = 0.2,
    min_gap: float = 0.08,
    search_ms: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """在原始音频上细化 coarse envelope 的 onset 时间。

    返回 (refined_times_seconds, strengths)。
    coarse onset 来自 hop 网格，通常比真实瞬态早约一个 STFT 分析窗；
    因此在 ±search_ms 的原始音频窗口内找局部能量峰值，并用抛物线细分。
    """
    if env is None or len(env) < 8 or audio is None or len(audio) < 128:
        return np.array([]), np.array([])

    coarse_times = find_onset_times_from_env(
        env, sr, hop, threshold=threshold, min_gap=min_gap,
    )
    if not coarse_times:
        return np.array([]), np.array([])

    radius = max(8, int(search_ms / 1000.0 * sr))
    kernel = np.ones(5, dtype=np.float64) / 5.0
    n_audio = len(audio)
    out: List[float] = []
    strengths: List[float] = []

    for t in coarse_times:
        center = first_start + int(round(t * sr))
        rel = center - audio_start
        if rel < -radius or rel > n_audio + radius:
            continue
        lo = max(0, rel - radius)
        hi = min(n_audio, rel + radius + 2)
        if hi - lo < 5:
            continue
        seg = audio[lo:hi].astype(np.float64)
        sm = np.convolve(seg * seg, kernel, mode="same")
        idx = int(np.argmax(sm))
        delta = 0.0
        if 1 <= idx < len(sm) - 1:
            y0, y1, y2 = sm[idx - 1], sm[idx], sm[idx + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        refined_sample = audio_start + lo + idx + delta
        out.append(refined_sample / sr)
        strengths.append(float(sm[idx]))

    return np.asarray(out, dtype=np.float64), np.asarray(strengths, dtype=np.float64)


def refine_onset_times_from_ring(
    env: np.ndarray,
    first_start: int,
    engine: "OnsetEngine",
    audio_start: int,
    sr: int,
    hop: int,
    coarse_bpm: float,
    threshold: float = 0.2,
    min_gap: float = 0.08,
    search_ms: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """环形缓冲版本的子采样 onset 定位，避免每次都复制整段原始音频。"""
    if env is None or len(env) < 8:
        return np.array([]), np.array([])

    coarse_times = find_onset_times_from_env(
        env, sr, hop, threshold=threshold, min_gap=min_gap,
    )
    if not coarse_times:
        return np.array([]), np.array([])

    radius = max(8, int(search_ms / 1000.0 * sr))
    kernel = np.ones(5, dtype=np.float64) / 5.0
    audio = engine.audio
    cap = engine.audio_cap
    total = engine.audio_total
    out: List[float] = []
    strengths: List[float] = []

    for t in coarse_times:
        center = first_start + int(round(t * sr))
        if center - radius < audio_start or center + radius >= total:
            continue
        lo = max(audio_start, center - radius)
        hi = min(total, center + radius + 2)
        if hi - lo < 5:
            continue
        idx = (np.arange(lo, hi, dtype=np.int64) % cap)
        seg = audio[idx].astype(np.float64)
        sm = np.convolve(seg * seg, kernel, mode="same")
        i = int(np.argmax(sm))
        delta = 0.0
        if 1 <= i < len(sm) - 1:
            y0, y1, y2 = sm[i - 1], sm[i], sm[i + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        refined_sample = lo + i + delta
        out.append(refined_sample / sr)
        strengths.append(float(sm[i]))

    return np.asarray(out, dtype=np.float64), np.asarray(strengths, dtype=np.float64)


def weighted_period_regression(
    times: np.ndarray,
    coarse_bpm: float,
    weights: Optional[np.ndarray] = None,
    min_beats: int = 6,
    residual_ratio: float = 0.15,
    max_iters: int = 4,
) -> Optional[Tuple[float, float, int, float, float]]:
    """加权最小二乘拟合 t_i = a + k_i*T + d*k_i^2。

    返回 (bpm, sigma_bpm, beats, residual_std_seconds, period_seconds)。
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(times) < min_beats:
        return None
    if weights is None or len(weights) != len(times):
        weights = np.ones(len(times), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights = np.maximum(weights, 1e-12)

    T = 60.0 / float(coarse_bpm)
    t0 = float(times[0])
    arr = times.copy()
    w = weights.copy()

    for _ in range(max_iters):
        if len(arr) < min_beats:
            return None
        k = np.round((arr - t0) / T)
        A = np.column_stack([np.ones_like(k), k, k * k])
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], arr * sw, rcond=None)
        pred = A @ coef
        resid = arr - pred
        T = float(coef[1])
        t0 = float(coef[0])

        keep = np.abs(resid) < residual_ratio * T
        if keep.sum() < min_beats:
            # 保留误差较小的若干点，避免一次淘汰过多导致无法继续
            order = np.argsort(np.abs(resid))
            keep_idx = order[:max(min_beats, int(len(arr) * 0.6))]
            arr = arr[keep_idx]
            w = w[keep_idx]
        elif keep.sum() < len(arr):
            arr = arr[keep]
            w = w[keep]
        else:
            break

    if len(arr) < min_beats:
        return None

    k = np.round((arr - t0) / T)
    A = np.column_stack([np.ones_like(k), k, k * k])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], arr * sw, rcond=None)
    pred = A @ coef
    resid = arr - pred
    dof = len(arr) - 3
    if dof <= 0:
        return None

    rss = float(np.sum(w * resid * resid))
    s2 = rss / dof
    normal = np.dot(A.T * w, A)
    try:
        cov = s2 * np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    var_T = float(cov[1, 1])
    if var_T < 0.0:
        return None

    sigma_T = float(np.sqrt(var_T))
    period = float(coef[1])
    bpm = 60.0 / period
    sigma_bpm = 60.0 * sigma_T / (period * period)
    return bpm, max(0.0, sigma_bpm), len(arr), float(np.std(resid)), period


# ---------------------------------------------------------------------------
# 策略基类
# ---------------------------------------------------------------------------

def _with_method(schema: List[dict]) -> List[dict]:
    return ([{"key": "method", "label": "Onset 方法", "type": "combo",
              "values": ["flux", "wflux", "rms"], "default": "flux"}] + schema)


class BaseStrategy:
    key = "base"
    display_name = "基础"
    description = ""
    settings_schema: List[dict] = []
    default_update_interval = 1.0
    method_field = {"key": "method", "label": "Onset 方法", "type": "combo",
                    "values": ["flux", "wflux", "rms"], "default": "flux"}

    def __init_subclass__(cls, **kwargs):
        """自动把 v3.0 精度字段填到各策略 update() 返回的 BpmResult 中。"""
        super().__init_subclass__(**kwargs)
        if "update" in cls.__dict__:
            original_update = cls.__dict__["update"]

            def wrapped_update(self, *args, **kwargs):
                result = original_update(self, *args, **kwargs)
                if result is not None and result.bpm is not None:
                    beats = getattr(self, "_beats", 0)
                    if beats > 0:
                        result.sigma_bpm = getattr(self, "_sigma_bpm", 0.0)
                        result.beats = beats
                return result

            cls.update = wrapped_update

    def __init__(self, settings: Optional[dict] = None,
                 audio_settings: Optional[dict] = None):
        self.settings = dict(settings or {})
        self.audio_settings = dict(audio_settings or {})
        self.sr = int(self.audio_settings.get("sample_rate", 22050))
        self.fft_size = int(self.audio_settings.get("fft_size", 1024))
        self.hop = int(self.audio_settings.get("hop_size", 512))
        # 实际分析使用更细的 hop，保证 BPM 精度
        self.analysis_hop = min(self.hop, 128)
        self.engine = OnsetEngine(self.sr, self.fft_size, self.analysis_hop,
                                  method=str(self.settings.get("method", "flux")))
        self.last_result = BpmResult(mode=self.display_name)
        self.last_update_time = 0.0
        self.reset()

    def reset(self) -> None:
        self.engine.reset()
        self.last_bpm: Optional[float] = None
        self.raw_bpm: Optional[float] = None
        self.confidence = 0.0
        self.status = "等待音频"
        self.last_result = BpmResult(mode=self.display_name, status=self.status)
        self.last_update_time = 0.0
        self.proc_ms = 0.0
        self._precision_bpm: Optional[float] = None
        self._sigma_bpm: float = 0.0
        self._beats: int = 0
        self._precision_residual: float = 0.0

    def push_audio(self, mono: np.ndarray) -> None:
        if mono is None or len(mono) == 0:
            return
        self.engine.push(np.asarray(mono, dtype=np.float32).reshape(-1))

    def get_update_interval(self) -> float:
        return float(self.settings.get("update_interval", self.default_update_interval))

    def process_audio(self, mono: np.ndarray,
                      sample_rate: Optional[int] = None) -> Optional[BpmResult]:
        """增量推入音频；到更新间隔时执行 update 并返回新结果，否则返回 None。

        v2.0 关键改动：未到更新间隔返回 None（不再返回缓存结果刷 UI）。
        """
        if sample_rate and sample_rate != self.sr:
            self.sr = int(sample_rate)
        t0 = time.perf_counter()
        self.push_audio(mono)
        now = time.monotonic()
        if now - self.last_update_time < self.get_update_interval():
            return None
        self.last_update_time = now
        result = self.update()
        result.proc_ms = (time.perf_counter() - t0) * 1000.0
        if result.bpm is not None and self._beats > 0:
            result.sigma_bpm = self._sigma_bpm
            result.beats = self._beats
        self.last_result = result
        return result

    def update(self) -> BpmResult:
        raise NotImplementedError

    def _autocorr(self, seconds: float) -> Tuple[Optional[float], float]:
        env = self.engine.env_tail(seconds)
        min_bpm = float(self.settings.get("min_bpm", 60.0))
        max_bpm = float(self.settings.get("max_bpm", 200.0))
        bpm, conf = estimate_bpm_from_envelope(
            env, self.sr, self.analysis_hop,
            min_bpm=min_bpm, max_bpm=max_bpm, ref_bpm=self._ref_bpm())

        self._precision_bpm = None
        self._sigma_bpm = 0.0
        self._beats = 0
        self._precision_residual = 0.0
        if bpm is not None and seconds >= 4.0:
            refined = self._refine_bpm(bpm, seconds)
            if refined is not None:
                pb, sigma, beats, resid = refined
                if beats >= 6 and abs(pb - bpm) <= 1.0:
                    self._precision_bpm = pb
                    self._sigma_bpm = sigma
                    self._beats = beats
                    self._precision_residual = resid
                    bpm = pb
        return bpm, conf

    def _refine_bpm(self, coarse_bpm: float, seconds: float
                    ) -> Optional[Tuple[float, float, int, float]]:
        """v3.0：用原始音频子采样 onset + 加权周期回归提高 BPM 精度。"""
        try:
            prec = float(self.settings.get("precision_seconds", PRECISION_SECONDS))
        except Exception:
            prec = PRECISION_SECONDS
        prec = max(float(seconds), min(float(prec), float(BUFFER_SECONDS)))

        env, first_start = self.engine.env_tail_with_start(prec)
        if len(env) < 10:
            return None

        audio_start = max(0, self.engine.audio_total - int(prec * self.sr))
        max_bpm = float(self.settings.get("max_bpm", 200.0))
        period_ms = 60000.0 / max(1.0, float(coarse_bpm))
        search_ms = float(np.clip(period_ms * 0.25, 25.0, 45.0))
        min_gap = max(0.08, 60.0 / max(max_bpm, 1.0) / 2.0)
        threshold = float(self.settings.get("onset_threshold", 0.2))

        times, strengths = refine_onset_times_from_ring(
            env, first_start, self.engine, audio_start, self.sr, self.analysis_hop,
            coarse_bpm, threshold=threshold, min_gap=min_gap,
            search_ms=search_ms,
        )
        if len(times) < 6:
            return None

        fit = weighted_period_regression(
            times, coarse_bpm, weights=strengths,
            min_beats=6, residual_ratio=0.15,
        )
        if fit is None:
            return None
        bpm, sigma_bpm, beats, resid_std, period = fit
        return bpm, sigma_bpm, beats, resid_std

    def _ref_bpm(self) -> Optional[float]:
        """节拍连续性参考：各策略可覆盖。"""
        return self.last_bpm if self.last_bpm is not None else None

    def _normal_bpm(self, bpm: Optional[float]) -> Optional[float]:
        if bpm is None:
            return None
        min_bpm = float(self.settings.get("min_bpm", 60.0))
        max_bpm = float(self.settings.get("max_bpm", 200.0))
        if min_bpm >= max_bpm:
            return None
        # 用 0.5 BPM 容差，避免 60.0 被数值误差翻成 120.0
        while bpm < min_bpm - 0.5:
            bpm *= 2.0
        while bpm > max_bpm + 0.5:
            bpm /= 2.0
        return bpm


# ---------------------------------------------------------------------------
# 方案 1：流式节拍跟踪（IOI）
# ---------------------------------------------------------------------------

class StreamingIOIStrategy(BaseStrategy):
    key = "streaming_ioi"
    display_name = "1. 流式节拍跟踪"
    description = "根据最近的 onset 间隔快速估计 BPM，反应快。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "onset_threshold", "label": "Onset 阈值", "type": "scale",
         "min": 0.05, "max": 0.9, "resolution": 0.05, "default": 0.2},
        {"key": "recent_beats", "label": "使用最近节拍数", "type": "scale",
         "min": 3, "max": 10, "resolution": 1, "default": 5},
        {"key": "max_interval", "label": "最大拍间隔(秒)", "type": "scale",
         "min": 0.5, "max": 3.0, "resolution": 0.1, "default": 1.5},
    ])

    def update(self) -> BpmResult:
        threshold = float(self.settings.get("onset_threshold", 0.2))
        user_max_interval = float(self.settings.get("max_interval", 1.5))
        recent_beats = int(float(self.settings.get("recent_beats", 5)))
        min_bpm = float(self.settings.get("min_bpm", 60.0))
        max_bpm = float(self.settings.get("max_bpm", 200.0))
        # 低速音乐拍间隔较长，自动放宽最大拍间隔限制。
        max_interval = max(user_max_interval, 60.0 / max(min_bpm, 1.0) * 1.5)

        env = self.engine.env_tail(8.0)
        times = find_onset_times_from_env(
            env, self.sr, self.analysis_hop,
            threshold=threshold, min_gap=max(0.1, 60.0 / max_bpm / 2.0),
        )
        if len(times) < 2:
            return BpmResult(mode=self.display_name, status="等待节拍",
                             confidence=0.0)

        intervals = np.diff(np.asarray(times, dtype=np.float64))
        intervals = intervals[intervals <= max_interval]
        if len(intervals) == 0:
            return BpmResult(mode=self.display_name, status="节拍间隔过大",
                             confidence=0.0)

        # 优先选取落在用户 BPM 范围内的节拍间隔，避免被半速/倍速带偏。
        bpms = 60.0 / intervals
        in_range = bpms[(bpms >= min_bpm - 0.5) & (bpms <= max_bpm + 0.5)]
        if len(in_range) >= 1:
            bpm = float(np.median(in_range[-min(recent_beats, len(in_range)):]))
            used = len(in_range)
        else:
            recent = intervals[-min(recent_beats, len(intervals)):]
            bpm = 60.0 / float(np.median(recent))
            bpm = self._normal_bpm(bpm)
            used = len(recent)
        if bpm is None:
            return BpmResult(mode=self.display_name, status="无法估计")

        ioi_conf = min(1.0, used / float(recent_beats))
        # 流式 IOI 负责快速响应；有足够窗口时用自相关结果保证精度。
        ac_bpm, ac_conf = self._autocorr(8.0)
        if ac_bpm is not None:
            bpm = ac_bpm
            used = int(8.0)
            ioi_conf = ac_conf

        self.raw_bpm = bpm
        self.last_bpm = bpm
        self.confidence = ioi_conf
        self.status = "流式跟踪"
        return BpmResult(
            mode=self.display_name,
            bpm=bpm,
            raw_bpm=bpm,
            confidence=self.confidence,
            status=self.status,
            detail=f"使用最近 {used} 个节拍间隔",
        )


# ---------------------------------------------------------------------------
# 方案 2：两级快速估计 + 稳定锁定
# ---------------------------------------------------------------------------

class FastStableStrategy(BaseStrategy):
    key = "fast_stable"
    display_name = "2. 两级快速/稳定"
    description = "先用短窗给出快速估计，置信度够高后切换到稳定值。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "fast_window", "label": "快速窗口(秒)", "type": "scale",
         "min": 1, "max": 4, "resolution": 0.5, "default": 2.0},
        {"key": "stable_window", "label": "稳定窗口(秒)", "type": "scale",
         "min": 5, "max": 12, "resolution": 1, "default": 8},
        {"key": "lock_threshold", "label": "锁定阈值", "type": "scale",
         "min": 0.05, "max": 1.0, "resolution": 0.05, "default": 0.25},
        {"key": "relock_threshold", "label": "失锁重新锁定阈值", "type": "scale",
         "min": 0.05, "max": 1.0, "resolution": 0.05, "default": 0.15},
    ])

    def reset(self) -> None:
        super().reset()
        self.stable_est: Optional[float] = None

    def _ref_bpm(self) -> Optional[float]:
        if self.stable_est is not None:
            return self.stable_est
        return super()._ref_bpm()

    def update(self) -> BpmResult:
        fast_sec = float(self.settings.get("fast_window", 2.0))
        stable_sec = float(self.settings.get("stable_window", 8.0))
        lock = float(self.settings.get("lock_threshold", 0.25))
        relock = float(self.settings.get("relock_threshold", 0.15))

        fast_bpm, fast_conf = self._autocorr(fast_sec)
        stable_bpm, stable_conf = self._autocorr(stable_sec)

        if stable_bpm is not None and stable_conf >= lock:
            display_bpm = stable_bpm
            status = "已锁定"
            confidence = stable_conf
            self.stable_est = stable_bpm
        elif fast_bpm is not None:
            display_bpm = fast_bpm
            status = "锁定中"
            confidence = fast_conf
        else:
            return BpmResult(mode=self.display_name, status="等待更多音频")

        self.raw_bpm = fast_bpm if fast_bpm is not None else stable_bpm
        self.last_bpm = display_bpm
        self.confidence = confidence
        self.status = status
        return BpmResult(
            mode=self.display_name,
            bpm=display_bpm,
            raw_bpm=self.raw_bpm,
            fast_bpm=fast_bpm,
            stable_bpm=stable_bpm,
            confidence=confidence,
            status=status,
            window_seconds=stable_sec if status == "已锁定" else fast_sec,
        )


# ---------------------------------------------------------------------------
# 方案 3：自适应滑动窗口
# ---------------------------------------------------------------------------

class AdaptiveWindowStrategy(BaseStrategy):
    key = "adaptive_window"
    display_name = "3. 自适应滑动窗口"
    description = "初期用短窗快速出结果，稳定后自动加长，发现变化后缩短。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "min_window", "label": "最短窗口(秒)", "type": "scale",
         "min": 2, "max": 8, "resolution": 0.5, "default": 5.0},
        {"key": "max_window", "label": "最长窗口(秒)", "type": "scale",
         "min": 6, "max": 12, "resolution": 1, "default": 8},
        {"key": "step", "label": "窗口调整步长", "type": "scale",
         "min": 0.5, "max": 2.0, "resolution": 0.5, "default": 0.5},
        {"key": "stability_threshold", "label": "稳定性阈值", "type": "scale",
         "min": 0.05, "max": 1.0, "resolution": 0.05, "default": 0.2},
        {"key": "jump_threshold", "label": "跳变阈值(BPM)", "type": "scale",
         "min": 1, "max": 10, "resolution": 1, "default": 3},
    ])

    def reset(self) -> None:
        super().reset()
        self.current_window = float(self.settings.get("min_window", 5.0))

    def update(self) -> BpmResult:
        min_win = float(self.settings.get("min_window", 2.0))
        max_win = float(self.settings.get("max_window", 8.0))
        step = float(self.settings.get("step", 0.5))
        stability = float(self.settings.get("stability_threshold", 0.2))
        jump = float(self.settings.get("jump_threshold", 3.0))

        if not hasattr(self, "current_window"):
            self.current_window = min_win

        bpm, conf = self._autocorr(self.current_window)
        if bpm is None:
            self.current_window = min_win
            return BpmResult(mode=self.display_name, status="窗口内数据不足",
                             window_seconds=self.current_window)

        if self.last_bpm is not None and abs(bpm - self.last_bpm) > jump:
            self.current_window = min_win
        elif conf >= stability:
            self.current_window = min(max_win, self.current_window + step)
        else:
            self.current_window = max(min_win, self.current_window - step)

        self.raw_bpm = bpm
        self.last_bpm = bpm
        self.confidence = conf
        self.status = "自适应窗口"
        return BpmResult(
            mode=self.display_name,
            bpm=bpm,
            raw_bpm=bpm,
            confidence=conf,
            status=self.status,
            window_seconds=self.current_window,
            detail=f"当前窗口 {self.current_window:.1f}s",
        )


# ---------------------------------------------------------------------------
# 方案 4：重叠式更新
# ---------------------------------------------------------------------------

class OverlapUpdateStrategy(BaseStrategy):
    key = "overlap_update"
    display_name = "4. 重叠式更新"
    description = "使用较长窗口，但高频滑动更新，让显示更连续。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "window", "label": "分析窗口(秒)", "type": "scale",
         "min": 4, "max": 12, "resolution": 1, "default": 8},
        {"key": "overlap_percent", "label": "重叠率(%)", "type": "scale",
         "min": 50, "max": 95, "resolution": 5, "default": 90},
    ])

    def get_update_interval(self) -> float:
        window = float(self.settings.get("window", 8.0))
        overlap = float(self.settings.get("overlap_percent", 90.0)) / 100.0
        interval = window * (1.0 - overlap)
        return max(0.1, interval)

    def update(self) -> BpmResult:
        window = float(self.settings.get("window", 8.0))
        bpm, conf = self._autocorr(window)
        if bpm is None:
            return BpmResult(mode=self.display_name, status="等待更多音频",
                             window_seconds=window)

        self.raw_bpm = bpm
        self.last_bpm = bpm
        self.confidence = conf
        self.status = "重叠更新"
        interval = self.get_update_interval()
        return BpmResult(
            mode=self.display_name,
            bpm=bpm,
            raw_bpm=bpm,
            confidence=conf,
            status=self.status,
            window_seconds=window,
            detail=f"每 {interval:.2f}s 更新一次",
        )


# ---------------------------------------------------------------------------
# 方案 5：自适应平滑
# ---------------------------------------------------------------------------

class AdaptiveSmoothingStrategy(BaseStrategy):
    key = "adaptive_smoothing"
    display_name = "5. 自适应平滑"
    description = "稳定时慢速平滑，突变时快速跟随。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "window", "label": "检测窗口(秒)", "type": "scale",
         "min": 4, "max": 12, "resolution": 1, "default": 8},
        {"key": "slow_alpha", "label": "慢速平滑系数", "type": "scale",
         "min": 0.01, "max": 0.3, "resolution": 0.01, "default": 0.05},
        {"key": "fast_alpha", "label": "快速平滑系数", "type": "scale",
         "min": 0.1, "max": 0.9, "resolution": 0.1, "default": 0.5},
        {"key": "change_threshold", "label": "变化阈值(BPM)", "type": "scale",
         "min": 1, "max": 10, "resolution": 1, "default": 3},
        {"key": "history_size", "label": "历史长度", "type": "scale",
         "min": 3, "max": 20, "resolution": 1, "default": 8},
    ])

    def reset(self) -> None:
        super().reset()
        self.smoothed_bpm: Optional[float] = None
        self.history: List[float] = []

    def _ref_bpm(self) -> Optional[float]:
        if self.smoothed_bpm is not None:
            return self.smoothed_bpm
        return super()._ref_bpm()

    def update(self) -> BpmResult:
        window = float(self.settings.get("window", 8.0))
        raw_bpm, conf = self._autocorr(window)
        if raw_bpm is None:
            return BpmResult(mode=self.display_name, status="等待更多音频",
                             window_seconds=window)

        slow_alpha = float(self.settings.get("slow_alpha", 0.05))
        fast_alpha = float(self.settings.get("fast_alpha", 0.5))
        change_threshold = float(self.settings.get("change_threshold", 3.0))
        history_size = int(float(self.settings.get("history_size", 8)))

        if self.smoothed_bpm is None:
            self.smoothed_bpm = raw_bpm
        else:
            if abs(raw_bpm - self.smoothed_bpm) >= change_threshold:
                alpha = fast_alpha
            else:
                alpha = slow_alpha
            self.smoothed_bpm = alpha * raw_bpm + (1.0 - alpha) * self.smoothed_bpm

        self.history.append(self.smoothed_bpm)
        if len(self.history) > history_size:
            self.history = self.history[-history_size:]

        self.raw_bpm = raw_bpm
        self.last_bpm = self.smoothed_bpm
        self.confidence = conf
        self.status = "自适应平滑"
        return BpmResult(
            mode=self.display_name,
            bpm=self.smoothed_bpm,
            raw_bpm=raw_bpm,
            confidence=conf,
            status=self.status,
            window_seconds=window,
            detail=f"平滑后 {self.smoothed_bpm:.1f}",
        )


# ---------------------------------------------------------------------------
# 方案 6：候选 BPM 跟踪
# ---------------------------------------------------------------------------

class CandidateTrackingStrategy(BaseStrategy):
    key = "candidate_tracking"
    display_name = "6. 候选 BPM 跟踪"
    description = "同时维护多个候选 BPM，按新节拍证据加分/衰减。"
    default_update_interval = 0.5
    settings_schema = _with_method([
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 40, "max": 200, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "candidate_count", "label": "候选数量", "type": "scale",
         "min": 3, "max": 20, "resolution": 1, "default": 9},
        {"key": "candidate_step", "label": "候选粒度(BPM)", "type": "scale",
         "min": 0.5, "max": 2.0, "resolution": 0.5, "default": 1.0},
        {"key": "decay", "label": "候选衰减速度", "type": "scale",
         "min": 0.5, "max": 1.0, "resolution": 0.05, "default": 0.85},
        {"key": "allow_octave", "label": "允许半速/双速候选", "type": "check",
         "default": True},
        {"key": "show_candidates", "label": "显示候选列表", "type": "check",
         "default": True},
    ])

    def reset(self) -> None:
        super().reset()
        self.candidate_scores: dict = {}

    def update(self) -> BpmResult:
        bpm, conf = self._autocorr(8.0)
        if bpm is None:
            return BpmResult(mode=self.display_name, status="等待更多音频")

        count = int(float(self.settings.get("candidate_count", 9)))
        step = float(self.settings.get("candidate_step", 1.0))
        decay = float(self.settings.get("decay", 0.85))
        allow_octave = bool(self.settings.get("allow_octave", True))
        min_bpm = float(self.settings.get("min_bpm", 60.0))
        max_bpm = float(self.settings.get("max_bpm", 200.0))

        half = count // 2
        base_values = [bpm + (i - half) * step for i in range(count)]
        if allow_octave:
            base_values += [bpm / 2.0, bpm * 2.0]
        base_values = [v for v in base_values if min_bpm <= v <= max_bpm]

        for k in list(self.candidate_scores.keys()):
            self.candidate_scores[k] *= decay

        for c in base_values:
            c_rounded = round(c, 1)
            score = float(np.exp(-((c - bpm) / max(step, 0.01)) ** 2))
            self.candidate_scores[c_rounded] = self.candidate_scores.get(c_rounded, 0.0) + score

        ranked = sorted(self.candidate_scores.items(), key=lambda x: x[1], reverse=True)
        self.candidate_scores = dict(ranked[:count])

        if not ranked:
            return BpmResult(mode=self.display_name, status="暂无候选")

        best_bpm, best_score = ranked[0]
        show_candidates = bool(self.settings.get("show_candidates", True))
        candidates = [(float(k), float(v)) for k, v in ranked[:5]] if show_candidates else []

        # v3.0：候选机制继续维护跟踪稳定性，但展示值使用细化后的高精度 BPM。
        display_bpm = bpm if self._beats > 0 else best_bpm
        self.raw_bpm = bpm
        self.last_bpm = display_bpm
        self.confidence = conf
        self.status = "候选跟踪"
        return BpmResult(
            mode=self.display_name,
            bpm=display_bpm,
            raw_bpm=bpm,
            confidence=conf,
            status=self.status,
            candidates=candidates,
            detail=f"最佳候选 {best_bpm:.3f}，得分 {best_score:.2f}",
        )


# ---------------------------------------------------------------------------
# 方案 7：用户限定 BPM 范围加速锁定
# ---------------------------------------------------------------------------

class BpmRangeStrategy(BaseStrategy):
    key = "bpm_range"
    display_name = "7. BPM 范围加速"
    description = "利用用户给定的 BPM 范围缩小搜索，快速锁定。"
    default_update_interval = 0.4
    PRESET_RANGES = {
        "自定义": None,
        "Hip-hop (70-110)": (70, 110),
        "电子 (120-140)": (120, 140),
        "流行 (90-130)": (90, 130),
        "摇滚 (100-160)": (100, 160),
    }
    settings_schema = _with_method([
        {"key": "preset", "label": "风格预设", "type": "combo",
         "values": ["自定义", "Hip-hop (70-110)", "电子 (120-140)",
                    "流行 (90-130)", "摇滚 (100-160)"],
         "default": "自定义"},
        {"key": "min_bpm", "label": "最小 BPM", "type": "scale",
         "min": 30, "max": 140, "resolution": 1, "default": 60},
        {"key": "max_bpm", "label": "最大 BPM", "type": "scale",
         "min": 80, "max": 400, "resolution": 1, "default": 200},
        {"key": "outside_mode", "label": "范围外处理", "type": "combo",
         "values": ["忽略", "折半", "加倍"], "default": "折半"},
        {"key": "fast_lock", "label": "快速锁定模式", "type": "check",
         "default": False},
    ])

    def update(self) -> BpmResult:
        preset = str(self.settings.get("preset", "自定义"))
        if preset in self.PRESET_RANGES and self.PRESET_RANGES[preset] is not None:
            lo, hi = self.PRESET_RANGES[preset]
            min_bpm, max_bpm = float(lo), float(hi)
        else:
            min_bpm = float(self.settings.get("min_bpm", 60.0))
            max_bpm = float(self.settings.get("max_bpm", 200.0))

        fast_lock = bool(self.settings.get("fast_lock", False))
        window = 5.0 if fast_lock else 8.0

        bpm, conf = self._autocorr(window)
        if bpm is not None:
            bpm = self._normal_bpm(bpm)
            outside_mode = str(self.settings.get("outside_mode", "折半"))
            if outside_mode == "忽略" and (bpm < min_bpm or bpm > max_bpm):
                bpm = None
            else:
                bpm = self._normal_bpm(bpm)

        if bpm is None:
            return BpmResult(mode=self.display_name, status="等待更多音频",
                             window_seconds=window)

        self.raw_bpm = bpm
        self.last_bpm = bpm
        self.confidence = conf
        self.status = "范围锁定中" if fast_lock else "范围跟踪"
        return BpmResult(
            mode=self.display_name,
            bpm=bpm,
            raw_bpm=bpm,
            confidence=conf,
            status=self.status,
            window_seconds=window,
            detail=f"范围 {min_bpm:.0f}-{max_bpm:.0f} BPM",
        )


# ---------------------------------------------------------------------------
# 策略注册表
# ---------------------------------------------------------------------------

STRATEGIES = [
    StreamingIOIStrategy,
    FastStableStrategy,
    AdaptiveWindowStrategy,
    OverlapUpdateStrategy,
    AdaptiveSmoothingStrategy,
    CandidateTrackingStrategy,
    BpmRangeStrategy,
]

STRATEGY_MAP = {cls.key: cls for cls in STRATEGIES}
