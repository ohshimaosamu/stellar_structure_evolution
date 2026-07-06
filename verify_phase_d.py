#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_phase_d.py — Phase D 検証スクリプト
  (1) 低温 opacity: F05 と GN93hz の重複域 (logT=3.75–4.5) での整合性、
      ブレンド後の κ(logT) が連続・単調に破綻しないこと
  (2) MLT 立方方程式: 効率/非効率極限、brentq 解との一致、
      弱対流近似との比較
"""
import importlib.util, sys, os
import numpy as np

spec = importlib.util.spec_from_file_location(
    "ph", os.path.join(os.path.dirname(__file__), "stellar_structure_phase6-10.py"))
ph = importlib.util.module_from_spec(spec)
sys.modules["ph"] = ph
spec.loader.exec_module(ph)

args = ph.build_default_args()
args.opal_X = 0.70
args.opal_Z = 0.02

print("=" * 70)
print("(1) 低温 opacity テーブルの検証")
print("=" * 70)

lt = ph._get_lowT_table(args)
assert lt is not None, "F05_lowT_g98.dat が見つかりません"
print(f"F05 grid: logT ∈ [{lt._logT_min:.2f}, {lt._logT_max:.2f}], "
      f"logR ∈ [{lt._logR_min:.1f}, {lt._logR_max:.1f}]")

opal = ph._get_opal_table('GN93hz.dat', 0.70, 0.02)
print(f"OPAL grid: logT ∈ [{opal._logT_min:.2f}, {opal._logT_max:.2f}]")

# 重複域での GN93(OPAL) vs GS98(F05) の比較 (logR 固定)
print("\n重複域 logT=3.80–4.50 での log10κ 比較 (太陽光球~エンベロープの logR)")
print(f"{'logT':>6} {'logR':>6} {'OPAL(GN93)':>11} {'F05(GS98)':>11} {'Δdex':>7}")
max_d = 0.0
for logR in (-1.0, -2.0, -3.0):
    for logT in (3.80, 3.90, 4.00, 4.10, 4.30, 4.50):
        T = 10.0**logT
        rho = 10.0**logR * (T / 1e6)**3
        ko = float(np.log10(opal.kappa_array(np.array([rho]), np.array([T]))[0]))
        kf = float(lt.logkappa(np.array([rho]), np.array([T]))[0])
        d = ko - kf
        max_d = max(max_d, abs(d))
        print(f"{logT:6.2f} {logR:6.1f} {ko:11.3f} {kf:11.3f} {d:7.3f}")
print(f"重複域の最大差: {max_d:.3f} dex")

# ブレンド後の連続性: 光球条件に近い (logR=-1.5) で logT を細かく掃引
print("\nブレンド後 κ(logT) の連続性チェック (logR=-1.5)")
logTs = np.linspace(3.5, 4.4, 181)
Ts = 10.0**logTs
rhos = 10.0**(-1.5) * (Ts / 1e6)**3
kap = ph.opacity(rhos, Ts, args)
lk = np.log10(kap)
dlk = np.abs(np.diff(lk))
print(f"max |Δlogκ| per 0.005 dex in logT: {dlk.max():.4f} "
      f"(at logT={logTs[np.argmax(dlk)]:.3f})")
assert dlk.max() < 0.15, "ブレンドで不連続なジャンプがあります"
print("→ 連続 (OK)")

# 太陽光球での κ の妥当性 (T=5777K, ρ≈2e-7 g/cm³ → κ ≈ 0.5–1 cm²/g 程度)
kph = float(ph.opacity(np.array([2.0e-7]), np.array([5777.0]), args)[0])
print(f"\n太陽光球 (T=5777K, ρ=2e-7): κ = {kph:.3f} cm²/g  (期待値 ~0.3–1)")

# 旧動作 (クリップ) との比較: logT=3.5 (M型光球相当)
kap_old = opal.kappa_array(np.array([1e-8]), np.array([10**3.5]))[0]
kap_new = float(ph.opacity(np.array([1e-8]), np.array([10**3.5]), args)[0])
print(f"logT=3.50, ρ=1e-8: 旧(クリップ) κ={kap_old:.3e} → 新(F05) κ={kap_new:.3e}"
      f"  (比 {kap_old/max(kap_new,1e-99):.1f}×)")

print("\n" + "=" * 70)
print("(2) 完全 MLT 立方方程式の検証")
print("=" * 70)

from scipy.optimize import brentq

def eta_brentq(U, W):
    if W <= 0:
        return 0.0
    hi = W / (np.sqrt(U*U + W) + U)
    f = lambda e: e**3 + (8.0*U/9.0)*(e*e + 2.0*U*e - W)
    return brentq(f, 0.0, hi*(1+1e-12) + 1e-300, xtol=1e-300, rtol=1e-15)

rng = np.random.default_rng(42)
Us = 10.0**rng.uniform(-8, 8, 4000)
Ws = 10.0**rng.uniform(-6, 6, 4000)
eta_v = ph._mlt_cubic_eta(Us, Ws)
rel = np.empty_like(eta_v)
for i, (u, w) in enumerate(zip(Us, Ws)):
    eb = eta_brentq(u, w)
    rel[i] = abs(eta_v[i] - eb) / max(eb, 1e-300)
print(f"brentq との相対差 (4000 乱数点, U∈[1e-8,1e8], W∈[1e-6,1e6]):")
print(f"  max = {rel.max():.2e}, median = {np.median(rel):.2e}")
assert rel.max() < 1e-10, "共通ソルバーが brentq と不一致"
print("→ 一致 (OK)")

# 極限の検証: ∇ = na + η(η+2U)
na0, W0 = 0.35, 0.10
nr0 = na0 + W0
for U in (1e-10, 1e-3, 1.0, 1e3, 1e10):
    eta = float(ph._mlt_cubic_eta(U, W0)[0])
    nab = na0 + eta*(eta + 2*U)
    print(f"U={U:9.1e}:  ∇={nab:.10f}   (∇_ad={na0}, ∇_rad={nr0})")
eta_lo = float(ph._mlt_cubic_eta(1e-10, W0)[0])
eta_hi = float(ph._mlt_cubic_eta(1e10, W0)[0])
assert abs(na0 + eta_lo*(eta_lo+2e-10) - na0) < 1e-6, "効率極限が ∇_ad に収束しない"
assert abs(na0 + eta_hi*(eta_hi+2e10) - nr0) < 1e-6, "非効率極限が ∇_rad に収束しない"
print("→ U→0 で ∇→∇_ad, U→∞ で ∇→∇_rad (OK)")

# 弱対流近似との比較 (v6 mlt_nabla の A/B): 太陽的な深部対流条件
print("\n弱対流近似 (旧) vs 完全 MLT (新) — 代表条件での ∇:")
class _A: pass
a2 = ph.build_default_args()
a2.opal_X, a2.opal_Z = 0.70, 0.02

conds = [
    # (説明, rho, T, P, r, m, L)  — 太陽エンベロープ相当
    ("CZ深部 (T=1e6)",   0.05, 1.0e6, 5.0e12, 0.75*ph.R_sun, 0.99*ph.M_sun, 0.72*ph.L_sun),
    ("CZ中部 (T=1e5)",   3e-4, 1.0e5, 2.0e10, 0.90*ph.R_sun, 0.999*ph.M_sun, 0.72*ph.L_sun),
    ("光球直下 (T=1e4)", 5e-7, 1.0e4, 5.0e6,  0.99*ph.R_sun, 1.0*ph.M_sun,  0.72*ph.L_sun),
]
print(f"{'条件':<16} {'∇_ad':>7} {'∇_rad':>9} {'∇(weak)':>9} {'∇(cubic)':>9}")
for name, rho, T, P, r, m, L in conds:
    rho_a = np.array([rho]); T_a = np.array([T]); P_a = np.array([P])
    kap = ph.opacity(rho_a, T_a, a2)
    na = ph.nabla_ad(rho_a, T_a, P_a, a2)
    nr = ph.nabla_rad(rho_a, T_a, P_a, np.array([L]), np.array([m]), kap)
    a2.mlt_solver = 'weak'
    nw, *_ = ph.mlt_nabla(nr, na, rho_a, T_a, P_a,
                          np.array([r]), np.array([m]), np.array([L]), kap, a2)
    a2.mlt_solver = 'cubic'
    nc, *_ = ph.mlt_nabla(nr, na, rho_a, T_a, P_a,
                          np.array([r]), np.array([m]), np.array([L]), kap, a2)
    print(f"{name:<16} {float(na[0]):7.4f} {float(nr[0]):9.4f} "
          f"{float(nw[0]):9.4f} {float(nc[0]):9.4f}")

print("\nすべての検証に合格しました。")
