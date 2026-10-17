import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dctn, fft2
from scipy.ndimage import gaussian_filter, convolve, maximum_filter
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(0)
N = 128
yy, xx = np.mgrid[:N, :N] / N
img = (np.sin(2*np.pi*(3*xx+2*yy)) * .3 + ((xx-.4)**2+(yy-.5)**2 < .05)*1.0 + ((np.abs(xx-.75) < .12) & (np.abs(yy-.3) < .1))*.7)
img += gaussian_filter(rng.standard_normal((N, N)), 1.2, mode="wrap") * .5
img = gaussian_filter(img, .8, mode="wrap")
roll = lambda a, s: np.roll(a, s, axis=(0, 1))
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)

# 1) FFT 크기 vs DCT 계수 : 1픽셀 이동 시 변화
shifts = [0, 1, 2, 3, 4, 8]
mf = [rel(np.abs(fft2(img)), np.abs(fft2(roll(img, (s, s))))) for s in shifts]
md = [rel(np.abs(dctn(img, norm="ortho")), np.abs(dctn(roll(img, (s, s)), norm="ortho"))) for s in shifts]
print("circular shift  FFT|.| change:", np.round(mf, 4), " DCT|.| change:", np.round(md, 3))
# 비순환 이동(오른쪽 위로 1px, 경계는 새로 채움)
def shift_cut(a, s): return np.pad(a, ((s, 0), (s, 0)), mode="edge")[:N, :N]
mf2 = [rel(np.abs(fft2(img)), np.abs(fft2(shift_cut(img, s)))) for s in shifts]
md2 = [rel(np.abs(dctn(img, norm="ortho")), np.abs(dctn(shift_cut(img, s), norm="ortho"))) for s in shifts]
print("non-circular    FFT:", np.round(mf2, 3), " DCT:", np.round(md2, 3))
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
ax[0, 0].imshow(img, cmap="gray"); ax[0, 0].set_title("원본")
ax[0, 1].imshow(np.log1p(np.abs(np.fft.fftshift(fft2(img)))), cmap="magma"); ax[0, 1].set_title("|FFT| (log)")
ax[0, 2].imshow(np.log1p(np.abs(dctn(img, norm="ortho")) * 20), cmap="magma"); ax[0, 2].set_title("|DCT| (log)")
d1 = roll(img, (1, 1))
ax[1, 0].imshow(np.abs(np.abs(fft2(img)) - np.abs(fft2(d1))), cmap="magma", vmin=0, vmax=1); ax[1, 0].set_title("||FFT| 차이| (1px 순환 이동): 0")
ax[1, 1].imshow(np.abs(np.abs(dctn(img, norm="ortho")) - np.abs(dctn(d1, norm="ortho"))), cmap="magma"); ax[1, 1].set_title(f"||DCT| 차이| (1px 이동): 상대 {md[1]:.2f}")
ax[1, 2].plot(shifts, md, "ro-", label="DCT 계수 크기"); ax[1, 2].plot(shifts, mf, "bs-", label="FFT 크기")
ax[1, 2].plot(shifts, md2, "r--", alpha=.5, label="DCT (비순환 이동)"); ax[1, 2].plot(shifts, mf2, "b--", alpha=.5, label="FFT (비순환 이동)")
ax[1, 2].set_xlabel("이동량 (px, 대각선)"); ax[1, 2].set_ylabel("상대 변화"); ax[1, 2].legend(); ax[1, 2].set_title("이동에 대한 변화")
for a in ax.ravel()[:5]: a.axis("off")
plt.tight_layout(); plt.savefig("figures/fft_vs_dct_shift.png", dpi=125); plt.close()

# 2) 에너지 압축 (DCT 가 더 압축적)
F = np.sort(np.abs(fft2(img)).ravel() ** 2)[::-1]; D = np.sort(np.abs(dctn(img, norm="ortho")).ravel() ** 2)[::-1]
Fcum, Dcum = np.cumsum(F) / F.sum(), np.cumsum(D) / D.sum()
ks = np.arange(1, 1001)
print("energy captured by top-k coef: k=100 FFT %.4f DCT %.4f | k=500 FFT %.4f DCT %.4f" % (Fcum[99], Dcum[99], Fcum[499], Dcum[499]))
fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(ks, 1 - Fcum[:1000], "b", label="FFT"); ax.plot(ks, 1 - Dcum[:1000], "r", label="DCT")
ax.set_yscale("log"); ax.set_xlabel("남긴 계수 수 k"); ax.set_ylabel("버려진 에너지 비율"); ax.legend(); ax.set_title("에너지 압축: DCT 가 더 빨리 감소 (이 영상)"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/energy_compaction.png", dpi=125); plt.close()

# 3) 합성곱 + 스트라이드 풀링은 이동 불변이 아니다 (고정 랜덤 필터, 학습 없음)
filters = [rng.standard_normal((5, 5)) for _ in range(8)]
def features(x, pool="max"):
    out = []
    for f in filters:
        y = np.maximum(convolve(x, f, mode="wrap"), 0)
        if pool == "max": y = maximum_filter(y, 2, mode="wrap")[::2, ::2]          # maxpool 2, stride 2
        elif pool == "blur":
            y = maximum_filter(y, 2, mode="wrap"); y = gaussian_filter(y, 1.0, mode="wrap")[::2, ::2]  # BlurPool(Zhang 2019)
        else: y = y                                                                  # 풀링 없음(전체 그대로)
        out.append(y)
    return np.stack(out)
def pooled_vec(x, pool):   # 마지막에 공간 평균(전역 풀링) → 분류기 입력 벡터
    return features(x, pool).mean((1, 2))
def equiv_err(pool, s):
    fa = features(img, pool); fb = features(roll(img, (s, s)), pool)
    cands = {s // 2, (s + 1) // 2}          # 스트라이드 2 특징맵이 맞출 수 있는 이동: 내림/올림
    return min(rel(np.roll(fa, (c, c), axis=(1, 2)), fb) for c in cands)
ss = range(0, 9)
E = {p: [equiv_err(p, s) for s in ss] for p in ["max", "blur"]}
print("equivariance err (best alignment) maxpool:", np.round(E["max"], 3), "blurpool:", np.round(E["blur"], 3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(list(ss), E["max"], "ro-", label="MaxPool (stride 2)"); ax[0].plot(list(ss), E["blur"], "bs-", label="블러 + stride 2")
ax[0].set_xlabel("이동량 (px, 대각선)"); ax[0].set_ylabel("특징맵 등변성 오차 (최적 정렬 후)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("짝수 이동은 정확히 맞고, 홀수(1px) 이동은 어긋남")
fa = features(img, "max")[0]; fb = features(roll(img, (1, 1)), "max")[0]
best = min([(rel(np.roll(fa, (c, c), (0, 1)), fb), c) for c in (0, 1)])[1]
ax[1].imshow(np.abs(np.roll(fa, (best, best), (0, 1)) - fb), cmap="magma"); ax[1].set_title("1px 이동 후 MaxPool 특징맵 차이 (밝을수록 큼)"); ax[1].axis("off")
plt.tight_layout(); plt.savefig("figures/stride_aliasing.png", dpi=125); plt.close()

# 4) 1D 설명: stride 2 샘플링은 위상(어느 샘플을 뽑느냐)에 의존
t = np.arange(32); x1 = np.sin(2*np.pi*.4*t)   # 0.4 cycles/px: 나이퀴스트(0.25 for stride2) 초과 → 에일리어싱
fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(t, x1, "gray", lw=1)
ax.plot(t[::2], x1[::2], "ro-", label="짝수 위치 샘플"); ax.plot(t[1::2], x1[1::2], "bs-", label="1px 이동 = 홀수 위치 샘플")
ax.legend(); ax.set_title("stride 2: 고주파(0.4 cycles/px)는 어느 위치를 뽑느냐에 따라 완전히 다른 결과 (에일리어싱)")
plt.tight_layout(); plt.savefig("figures/aliasing_1d.png", dpi=125); plt.close()
