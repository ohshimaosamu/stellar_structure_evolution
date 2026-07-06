#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finalize_solution.py — present-Sun のチェックポイント (ph6_fit_sol.npz) を
読み込み、組成プロファイルを再設定した上で診断とテーブル出力のみを行う。
(再ソルブは行わないので数秒で完了。ターン境界で本ソルブが中断されても
 第1段チェックポイントから最終成果物を得るための補助。)

使い方:
  python3 finalize_solution.py CHECKPOINT.npz modelS_Xq.dat OUT_table.txt [X_glob Z_glob]
"""
import sys, os
import importlib.util
import numpy as np

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ph6_fit_sol.npz"
prof = sys.argv[2] if len(sys.argv) > 2 else "none"
out  = sys.argv[3] if len(sys.argv) > 3 else "ph10_out_table.txt"
Xg   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.70
Zg   = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02

spec = importlib.util.spec_from_file_location(
    "ph", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "stellar_structure_phase6-10.py"))
ph = importlib.util.module_from_spec(spec)
sys.modules["ph"] = ph
spec.loader.exec_module(ph)

# 組成プロファイル or 一様組成
if prof.lower() == "none":
    ph.set_composition_profile(None)
    Xs, Zs = Xg, Zg
    ph.apply_composition(Xs, max(1.0 - Xs - Zs, 0.0), Zs)
    print(f"[finalize] 一様組成 X={Xs} Z={Zs}")
else:
    P = ph.load_composition_profile(prof)
    Xs = float(P['X'][-1]); Zs = float(P['Z'][-1])
    ph.apply_composition(Xs, max(1.0 - Xs - Zs, 0.0), Zs)

args = ph.build_default_args()
args.mlt_alpha = 1.8
args.opal_X = Xs; args.opal_Z = Zs

d = np.load(ckpt, allow_pickle=True)
Xsol = d["X"]; q = d["q"]
M_hat = float(d["M_hat"]) if "M_hat" in d else 1.0
ph.M_hat = M_hat
N = len(q)

# sol オブジェクトを最小限で模倣 (write_table7 は sol.x, sol.fun を使う)
class _Sol:
    pass
sol = _Sol()
sol.x = Xsol
# 残差を再評価 (tab は診断に不要なので簡易に); write_table7 が sol.fun を使うなら 0 埋め
sol.fun = np.zeros(1)

# 診断出力
S, ell, p, t, lLe = ph.unpack6(Xsol[:-1], N)
lR = float(Xsol[-1]); lL = float(lLe)
T_c = ph.T_REF * np.exp(t[0]); P_c = ph.P_REF * np.exp(p[0])
_Xc, _Yc, _Zc, _muc, _XCNOc = ph._comp_at(np.array([q[0]]))
rho_c = float(ph.density_from_PT(np.array([P_c]), np.array([T_c]), mu_loc=_muc)[0])
env = ph.integrate_envelope(np.exp(lR), np.exp(lL), args, fit_dm=1e-4)
Xn, Yn, Zn, mun, XCNOn = ph._comp_at(q)
P_n = ph.P_REF * np.exp(p); T_n = ph.T_REF * np.exp(t)
rho_n = ph.density_from_PT(P_n, T_n, mu_loc=mun)
eps_n = ph.energy_generation(rho_n, T_n, X_loc=Xn, XCNO_loc=XCNOn)
L_mass = float(np.sum(0.5*(eps_n[:-1]+eps_n[1:])*np.diff(q))) * M_hat * ph.M_sun
L_star = ph.L_sun * np.exp(lL) * float(ell[-1])
print("="*60)
print(f"solution ({prof}):")
print(f"  R={np.exp(lR):.4f} R☉  L={L_star/ph.L_sun:.4f} L☉  "
      f"T_eff={env['T_eff']:.0f} K" if env else "")
print(f"  T_c={T_c/1e6:.3f} MK  ρ_c={rho_c:.2f} g/cm³  "
      f"∫εdm/L={L_mass/max(L_star,1e-9):.4f}")

qm = 0.5*(q[:-1]+q[1:])
nab, nrv, nav = ph.choose_nabla7(qm, 0.5*(S[:-1]+S[1:]), 0.5*(p[:-1]+p[1:]),
                                 0.5*(ell[:-1]+ell[1:]), 0.5*(t[:-1]+t[1:]),
                                 lL, args)
print(f"  中心: ∇_rad={nrv[0]:.3f} / ∇_ad={nav[0]:.3f} "
      f"({'対流コア' if nrv[0]>nav[0] else '放射的'})")
kk = np.where((nrv > nav) & (qm > 0.5))[0]
if len(kk):
    k0 = kk[0]
    r_bcz = np.exp(0.5*(S[k0]+S[k0+1]))
    print(f"  表面対流層底: q={qm[k0]:.5f}  r={r_bcz:.4f} R☉  "
          f"r/R*={r_bcz/np.exp(lR):.4f}  "
          f"T={ph.T_REF*np.exp(0.5*(t[k0]+t[k0+1])):.3e} K")

# テーブル出力
ph.write_table7(sol, q, args, mode='l', outfile=out, fit_dm=1e-4)
print(f"テーブル出力: {out}")
