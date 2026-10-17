"""위상 기반 프레임 보간 장난감 구현 (Meyer et al. 2015/PhaseNet 2018 의 핵심 원리).
복소 대역통과 분해 → 대역별 위상 차이의 절반만큼 이동 → 합성. 큰 이동은 거친 스케일의 위상으로 보정."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(7)
H, W = 96, 256
fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1))

# 옥타브 대역 (수평 주파수 기준, 한쪽(+fx)만 → 해석신호)
fcs = [0.25 * 2.0 ** (-k) for k in range(5)]            # 0.25 … 0.0156
def band(fc):
    r = np.log2(np.maximum(fx, 1e-9) / fc)
    return np.where((fx > 0) & (np.abs(r) < 1), np.cos(np.pi / 2 * r) ** 2, 0.0)
bands = [band(fc) for fc in fcs]
# 합이 1 이 되도록 정규화(한쪽 스펙트럼 기준); 나머지(저주파 잔차)는 별도
tot = sum(bands) + 1e-12
def decompose(img):
    F = np.fft.fft2(img)
    coef = [np.fft.ifft2(F * b) for b in bands]                 # 복소 대역 신호
    covered = sum(bands) + sum(b[:, ::-1] for b in bands) * 0  # (표시용)
    return coef, F
def reconstruct(coefs, F_ref):
    # 대역 합 (실수부*2) + 대역에 포함되지 않은 주파수는 기준 프레임 것을 사용
    out = np.zeros((H, W)); mask_in = np.zeros((H, W))
    for c, b in zip(coefs, bands):
        out += 2 * np.real(c)
    full = sum(bands); sym = full + np.roll(full[:, ::-1], 1, axis=1)   # 대역이 덮는 +/-fx
    resid = np.real(np.fft.ifft2(F_ref * (1 - np.minimum(sym, 1))))
    return out + resid
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def interp(I0, I1, t=0.5, cross_scale=True):
    c0, F0 = decompose(I0); c1, F1 = decompose(I1)
    dphi = [np.angle(b * np.conj(a)) for a, b in zip(c0, c1)]
    if cross_scale:   # 거친 대역 → 미세 대역 순으로 위상차 보정 (주파수 2배 → 이동에 따른 위상차 2배)
        # 대역 인덱스 k=0 이 가장 미세. k 가 커질수록 거침
        for k in range(len(fcs) - 2, -1, -1):
            pred = 2 * dphi[k + 1]
            dphi[k] = pred + wrap(dphi[k] - pred)
    out = []
    for a, b, d in zip(c0, c1, dphi):
        mag = (1 - t) * np.abs(a) + t * np.abs(b)
        out.append(mag * np.exp(1j * (np.angle(a) + t * d)))
    return reconstruct(out, np.fft.fft2(I0) * (1 - t) + np.fft.fft2(I1) * t)

# 테스트 장면: 정지한 질감 배경 + 이동하는 질감 사각형
def texture(seed, s=1.2):
    r = np.random.default_rng(seed); x = gaussian_filter(r.standard_normal((H, W)), s, mode="wrap"); return (x - x.min()) / (x.max() - x.min())
bg = texture(1, 1.5); fgt = texture(2, 1.0)
def scene(shift):
    yy, xx = np.mgrid[:H, :W]
    s = shift
    x0 = 90 + s
    img = bg.copy()
    # 사각형 내부 질감이 사각형과 함께 이동 (서브픽셀은 푸리에 이동)
    sh = lambda a, d: np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2j * np.pi * np.fft.fftfreq(W)[None, :] * d)))
    tex = sh(fgt, s)
    m = sh(((xx >= 90) & (xx < 150) & (yy > 25) & (yy < 70)).astype(float), s)
    m = np.clip(m, 0, 1)
    return img * (1 - m) + tex * m
CR = (slice(20, 75), slice(80, 176))   # 움직이는 영역
psnr = lambda a, b: 10 * np.log10(1 / np.mean((a[CR] - b[CR]) ** 2))
rows = []
fig, ax = plt.subplots(3, 4, figsize=(17, 7))
for r, d in enumerate([1.5, 6.0, 14.0]):
    I0, I1, GT = scene(0), scene(d), scene(d / 2)
    lin = (I0 + I1) / 2
    ph0 = interp(I0, I1, .5, cross_scale=False); ph = interp(I0, I1, .5, cross_scale=True)
    res = [("선형 혼합(블렌딩)", lin), ("위상 보간 (스케일 보정 없음)", ph0), ("위상 보간 (거친→미세 보정)", ph)]
    rows.append((d, [psnr(np.clip(v, 0, 1), GT) for _, v in res]))
    ax[r, 0].imshow(GT[CR], cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"정답 중간 프레임 (이동 {d:g}px의 절반)")
    for c, (nm, v) in enumerate(res, 1):
        ax[r, c].imshow(np.clip(v, 0, 1)[CR], cmap="gray", vmin=0, vmax=1); ax[r, c].set_title(f"{nm}\n움직이는 영역 PSNR {psnr(np.clip(v,0,1),GT):.1f} dB", fontsize=10)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig("figures/interp_compare.png", dpi=120); plt.close()
for d, p in rows: print(d, [round(x, 2) for x in p])

# 위상 = 움직임: 1D 예시
x = np.arange(256); f = lambda s: np.exp(-((x - 100 - s) / 6) ** 2) + .5 * np.sin(2 * np.pi * .125 * (x - s)) * np.exp(-((x - 100 - s) / 30) ** 2)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for s, c in [(0, "b"), (6, "r")]: ax[0].plot(x, f(s), c, label=f"프레임 shift={s}")
ax[0].plot(x, (f(0) + f(6)) / 2, "g--", label="선형 혼합"); ax[0].plot(x, f(3), "k:", label="정답(중간)"); ax[0].legend(fontsize=8); ax[0].set_xlim(60, 160); ax[0].set_title("공간 영역: 혼합은 유령 이중상")
F0, F1 = np.fft.rfft(f(0)), np.fft.rfft(f(6)); fr = np.fft.rfftfreq(256)
ax[1].plot(fr[:60], np.abs(F0[:60]), "b"); ax[1].plot(fr[:60], np.abs(F1[:60]), "r--"); ax[1].set_title("크기 |F| : 이동해도 동일"); ax[1].set_xlabel("주파수")
ph = np.angle(F1 * np.conj(F0)); sel = np.abs(F0) > 1e-3
ax[2].plot(fr[sel][:40], ph[sel][:40], "o", ms=3); ax[2].plot(fr[:60], wrap(-2 * np.pi * fr[:60] * 6), "r-", lw=.8, label="−2πf·6 (감김)"); ax[2].legend(); ax[2].set_title("위상차 = −2πf·d : 주파수에 비례"); ax[2].set_xlabel("주파수")
plt.tight_layout(); plt.savefig("figures/phase_is_motion.png", dpi=125); plt.close()
