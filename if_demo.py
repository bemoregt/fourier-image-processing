import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft, savgol_filter
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(0)
fs = 1000; T = 2.0; t = np.arange(0, T, 1/fs)

def inst_freq_diff(z):          # 위상 '차분': angle(z[n+1]·conj(z[n]))  → 언래핑 불필요
    return np.angle(z[1:] * np.conj(z[:-1])) * fs / (2*np.pi)
def inst_freq_unwrap(z):        # 언래핑 후 미분
    return np.gradient(np.unwrap(np.angle(z)), 1/fs) / (2*np.pi)

# 1) 선형 처프 + 사인 FM
f_true = 50 + 100*t + 30*np.sin(2*np.pi*1.0*t)          # Hz
phase = 2*np.pi*np.cumsum(f_true)/fs
x = np.cos(phase)
z = hilbert(x)
fi_d = inst_freq_diff(z); fi_u = inst_freq_unwrap(z)
sl = slice(100, -100)       # 힐베르트 변환 경계 효과 제외
print("clean: RMS err diff %.3f Hz, unwrap %.3f Hz" % (np.sqrt(np.mean((fi_d[99:-100]-f_true[1:][99:-100])**2)), np.sqrt(np.mean((fi_u[sl]-f_true[sl])**2))))
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax[0, 0].plot(t, x, lw=.6); ax[0, 0].set_xlim(0, .6); ax[0, 0].set_title("신호 x(t): 처프 + 정현파 FM")
ph = np.angle(z)
ax[0, 1].plot(t[:400], ph[:400], "gray", lw=.8, label="위상 arg z(t) (감김, −π~π)"); ax[0, 1].plot(t[:400], np.unwrap(ph)[:400] - np.unwrap(ph)[0], "r", label="언래핑한 위상 (누적)"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("해석신호의 위상")
ax[1, 0].plot(t, f_true, "k", lw=2, label="실제 f(t)"); ax[1, 0].plot(t[1:][sl], fi_d[sl], "r", lw=.8, label="위상 차분"); ax[1, 0].plot(t[sl], fi_u[sl], "b--", lw=.8, label="언래핑+미분")
ax[1, 0].legend(); ax[1, 0].set_title("순시주파수 f(t) = (1/2π)·dφ/dt"); ax[1, 0].set_xlabel("t (s)"); ax[1, 0].set_ylabel("Hz")
X = np.abs(np.fft.rfft(x)); ax[1, 1].plot(np.fft.rfftfreq(len(x), 1/fs), X/X.max()); ax[1, 1].set_xlim(0, 300); ax[1, 1].set_title("전체 FFT: 주파수가 '언제' 바뀌는지는 사라진다"); ax[1, 1].set_xlabel("Hz")
plt.tight_layout(); plt.savefig("figures/instantaneous_frequency.png", dpi=125); plt.close()

# 2) 잡음에 대한 민감도: 위상 미분은 잡음을 증폭
snrs = [40, 30, 20, 10, 5]
errs = {k: [] for k in ["raw diff", "smoothed diff"]}
for snr in snrs:
    e = {k: [] for k in errs}
    for _ in range(10):
        xn = x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x**2) / 10**(snr/10))
        zn = hilbert(xn); d = inst_freq_diff(zn); ft = f_true[1:]
        e["raw diff"].append(np.sqrt(np.mean((d[99:-100] - ft[99:-100])**2)))
        ds = savgol_filter(d, 51, 2); e["smoothed diff"].append(np.sqrt(np.mean((ds[99:-100] - ft[99:-100])**2)))
    for k in errs: errs[k].append(np.mean(e[k]))
print("SNR", snrs, {k: np.round(v, 2) for k, v in errs.items()})
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].loglog(snrs[::-1], errs["raw diff"][::-1], "ro-", label="위상 차분 (원시)"); ax[0].loglog(snrs[::-1], errs["smoothed diff"][::-1], "bs-", label="+ Savitzky–Golay 평활 (51점)")
ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("주파수 RMS 오차 (Hz)"); ax[0].legend(); ax[0].grid(alpha=.3, which="both"); ax[0].set_title("미분은 잡음을 증폭: 낮은 SNR 에서 오차 급증")
xn = x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x**2) / 10**(1.0)); zn = hilbert(xn); d = inst_freq_diff(zn)
ax[1].plot(t, f_true, "k", lw=2, label="실제"); ax[1].plot(t[1:][sl], d[sl], "r", lw=.5, alpha=.7, label="원시(SNR 10 dB)"); ax[1].plot(t[1:][sl], savgol_filter(d, 51, 2)[sl], "b", lw=1.2, label="평활")
ax[1].set_ylim(0, 300); ax[1].legend(fontsize=8); ax[1].set_title("SNR 10 dB 예시"); ax[1].set_xlabel("t (s)")
plt.tight_layout(); plt.savefig("figures/noise_sensitivity.png", dpi=125); plt.close()

# 3) 다중 성분: 순시주파수가 의미를 잃는 경우 (두 톤)
x2 = np.cos(2*np.pi*80*t) + np.cos(2*np.pi*120*t)
z2 = hilbert(x2); d2 = inst_freq_diff(z2); env2 = np.abs(z2)
print("two-tone: IF range %.1f..%.1f Hz, envelope min %.3f" % (np.percentile(d2[100:-100], 1), np.percentile(d2[100:-100], 99), env2[100:-100].min()))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
ax[0].plot(t, x2, lw=.6); ax[0].set_xlim(0, .3); ax[0].set_title("80 Hz + 120 Hz (두 성분)")
ax[1].plot(t[1:][sl], d2[sl], "r", lw=.7); ax[1].set_ylim(-500, 300); ax[1].set_title("위상 차분 순시주파수: 평균 100 Hz 부근, 포락선 최소점에서 음의 값까지 튐")
ax[2].plot(t[sl], env2[sl], "g"); ax[2].set_title("포락선 |z| : 0 근처에서 위상이 불안정"); ax[2].set_xlim(0, .3)
for a in ax: a.set_xlabel("t (s)")
plt.tight_layout(); plt.savefig("figures/multicomponent.png", dpi=125); plt.close()

# 4) STFT 와 비교 (해상도 trade-off) + FM 복조
f_, tt, Z = stft(x, fs, nperseg=128, noverlap=120)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].pcolormesh(tt, f_, np.abs(Z), shading="auto"); ax[0].plot(t, f_true, "w--", lw=1, label="실제 f(t)"); ax[0].set_ylim(0, 300); ax[0].legend(); ax[0].set_title("STFT(128점 창): 시간·주파수 해상도 trade-off로 번짐")
ax[1].plot(t, f_true, "k", lw=2, label="실제"); ax[1].plot(t[1:][sl], fi_d[sl], "r", lw=.8, label="위상 미분"); ax[1].set_ylim(0, 300); ax[1].legend(); ax[1].set_title("위상 미분: 샘플 단위 해상도 (단일 성분일 때)")
plt.tight_layout(); plt.savefig("figures/stft_vs_if.png", dpi=125); plt.close()
