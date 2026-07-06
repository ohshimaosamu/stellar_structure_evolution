#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stellar_structure_phase6.py  —  真の ZAMS 計算コード (Phase 6, Rev 3 + Phase D)
=======================================================================

【Phase D: 物理の精密化】(このリビジョンで追加)
  D-1. 低温 opacity テーブル (Ferguson et al. 2005, GS98 混合)
       - 同梱ファイル F05_lowT_g98.dat (logT: 2.70–4.50, logR: -8…+1)
       - GN93hz (logT≥3.75) と logT = 4.00±0.05 で smoothstep ブレンド
       - K 後期〜M 型の光球 (logT<3.75) がテーブル範囲内になり、
         旧実装のテーブル端クリップ (κ を logT=3.75 の値で代用) を解消
       - 混合組成の注意: OPAL 側は GN93、低温側は GS98。両者の κ 差は
         この温度域で数%であり、ブレンド帯の整合性は検証済み
       - フォールバック: ファイル不在時は従来動作 (クリップ + 解析 H⁻)
  D-2. 完全 MLT (K&W §7 立方方程式) の全面採用
       - 旧 v6 経路の弱対流近似 (Henyey 1965) を置き換え
       - 共通ソルバー _mlt_cubic_eta(): η³+(8U/9)(η²+2Uη−W)=0 の唯一の
         正根をベクトル化二分法 (64 回) + 単調性保証で解く
       - choose_nabla7 / integrate_envelope / v6 mlt_nabla の 3 経路を
         同一実装に統一 (brentq 点別呼び出しを廃止し高速化)
       - 検証: 効率極限 U→0 で ∇→∇_ad、非効率極限 U→∞ で ∇→∇_rad、
         brentq 解との一致 <1e-12 (verify_phase_d.py)
       - 表面境界 (surface_bc_values) の κ_phot も解析 H⁻ 近似から
         テーブル opacity へ統一

【設計思想】
  eps_scale を撤廃し、R と L を物理的に正しい固有値として求める。

【変数】
  S(q)   = log(r / R_sun)        増加列  R_star = exp(S[-1]) × R_sun (出力)
  ell(q) = L / L_est             増加列  ell[-1] ≈ 1 が L_est を拘束する
  p(q)   = log(P / P_REF)        減少列  P_REF = G M_sun² / (4π R_sun⁴)
  t(q)   = log(T / T_REF)        減少列  T_REF = G M_sun μ m_H / (k_B R_sun)
  log_L_est = log(L_est / L_sun) スカラー (グローバル未知数)

  L_star (出力) = L_est × ell[-1] × L_sun  ≈ L_est × L_sun (ell[-1]→1 のとき)

【核反応率係数 (文献値 Kippenhahn 2012)】
  pp:  2.4e6  (旧 Phase 4 の 2.4e4 は 100 倍過小)
  CNO: 8.24e25 (旧 Phase 4 の 8.7e27 は 100 倍過大)
  PP_CALIB / CNO_CALIB で太陽モデルを微調整する

【パック構造 (4N+1 変数)】
  X = [S_params(N), ell_params(N), p_params(N), t_params(N), log_L_est]

【残差 (4N-1 個, 過決定)】
  Rs    (N-1) : dS/dq   = M̂ ρ̄_s / (ρ exp(3S))
  Rp    (N-1) : dp/dq   = -M̂² q / exp(p+4S)
  Rell  (N-1) : dell/dq = M̂ ε / (EPS_REF × exp(log_L_est) × ell)
  Rt    (N-1) : dt/dq   = ∇ × dp/dq
  Rbc_t  (1)  : t[-1]  = log(T_eff(L_est×ell[-1], R_sun×exp(S[-1])) / T_REF)
  Rbc_p  (1)  : p[-1]  = log(P_phot(R_sun×exp(S[-1]), M_hat) / P_REF)
  Rbc_ell(1)  : ell[-1] - 1.0  ← log_L_est を拘束する
  Rcen_S (1)  : S[0] と中心級数展開の整合
  Rcen_ell(1) : ell[0] と中心光度の整合
"""

import argparse, sys, warnings, os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────
# 物理定数・太陽値
# ─────────────────────────────────────────────────────
G        = 6.674e-8
c_light  = 2.998e10
a_rad    = 7.566e-15
k_B      = 1.381e-16
m_H      = 1.673e-24
sigma_sb = a_rad * c_light / 4.0

M_sun = 1.989e33
R_sun = 6.960e10
L_sun = 3.828e33

# ─────────────────────────────────────────────────────
# Phase 6 固定参照スケール
# ─────────────────────────────────────────────────────
P_REF  = G * M_sun**2 / (4.0 * np.pi * R_sun**4)
RHO_S  = M_sun / (4.0 * np.pi * R_sun**3)
EPS_REF = L_sun / M_sun

# 組成 (apply_composition で更新)
X   = 0.70; Y = 0.28; Z = 0.02
mu  = 1.0 / (2*X + 0.75*Y + 0.5*Z)
T_REF = G * M_sun * mu * m_H / (k_B * R_sun)

M_hat = 1.0  # M_star / M_sun (current)

# 核反応率較正定数
PP_CALIB  = 1.0
CNO_CALIB = 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


def apply_composition(Xn, Yn, Zn):
    global X, Y, Z, mu, T_REF
    X = float(Xn); Y = float(Yn); Z = float(Zn)
    mu = 1.0 / max(2*X + 0.75*Y + 0.5*Z, 1e-99)
    T_REF = G * M_sun * mu * m_H / (k_B * R_sun)


# ══════════════════════════════════════════════════════
#  Phase E: 非一様組成プロファイル X(q)  (静的現在太陽モデル用)
# ══════════════════════════════════════════════════════
#  ZAMS は X,Y,Z がグローバル定数 (一様)。現在の太陽は 4.57 Gyr の核燃焼で
#  中心 X が枯渇した非一様プロファイル X(m) を持つ。ここでは文献 (Model S)
#  の X(q), Z(q) を外部入力として与え、静的に構造方程式を解く。
#  _COMP_PROFILE が None なら従来通り一様組成 (グローバル X,Y,Z) を使う。
_COMP_PROFILE = None   # dict(q[asc], X, Z, XCNO_frac) or None


def set_composition_profile(prof):
    """組成プロファイルを設定。prof=None で一様組成に戻す。
    prof: dict(q=昇順配列, X=配列, Z=配列, [XCNO_frac=金属中CNO質量比 既定0.7])"""
    global _COMP_PROFILE
    if prof is None:
        _COMP_PROFILE = None
        return
    q = np.asarray(prof['q'], float)
    idx = np.argsort(q)
    _COMP_PROFILE = dict(
        q=q[idx],
        X=np.asarray(prof['X'], float)[idx],
        Z=np.asarray(prof['Z'], float)[idx],
        XCNO_frac=float(prof.get('XCNO_frac', 0.7)))


def load_composition_profile(path, xcno_frac=0.7):
    """テキストファイル (列: q X Z) から組成プロファイルを読み込む。"""
    dat = np.loadtxt(path)
    set_composition_profile(dict(q=dat[:, 0], X=dat[:, 1],
                                 Z=dat[:, 2], XCNO_frac=xcno_frac))
    return _COMP_PROFILE


def _comp_at(q):
    """位置 q (スカラー or 配列) での (X, Y, Z, mu, X_CNO) を返す。
    プロファイル未設定なら一様グローバル値をブロードキャストして返す。"""
    if _COMP_PROFILE is None:
        qa = np.atleast_1d(np.asarray(q, float))
        one = np.ones_like(qa)
        return (X * one, Y * one, Z * one, mu * one, (Z / 2.0) * one)
    P = _COMP_PROFILE
    qa = np.atleast_1d(np.asarray(q, float))
    Xv = np.interp(qa, P['q'], P['X'])
    Zv = np.interp(qa, P['q'], P['Z'])
    Yv = np.maximum(1.0 - Xv - Zv, 0.0)
    muv = 1.0 / np.maximum(2.0 * Xv + 0.75 * Yv + 0.5 * Zv, 1e-99)
    XCNO = P['XCNO_frac'] * Zv     # 金属質量分率のうち CNO の割合
    return (Xv, Yv, Zv, muv, XCNO)


# ══════════════════════════════════════════════════════
# 1.  EOS / 核反応率
# ══════════════════════════════════════════════════════

def density_from_PT(P, T, floor=1e-20, mu_loc=None):
    T  = np.asarray(T, dtype=float)
    P  = np.asarray(P, dtype=float)
    mu_e = mu if mu_loc is None else mu_loc      # Phase E: 非一様 μ(q)
    rho = np.maximum(P * mu_e * m_H / (k_B * np.maximum(T, 1e3)), floor)
    for _ in range(6):
        Prad = a_rad * T**4 / 3.0
        rho  = np.maximum((P - Prad) * mu_e * m_H / (k_B * np.maximum(T, 1e3)), floor)
    return rho


def pressure_total(rho, T, mu_loc=None):
    mu_e = mu if mu_loc is None else mu_loc
    return rho * k_B * T / (mu_e * m_H) + a_rad * T**4 / 3.0


def energy_generation_components(rho, T, X_loc=None, XCNO_loc=None):
    """
    核エネルギー生成率 [erg g⁻¹ s⁻¹]。

    Phase E: X_loc (H 質量分率), XCNO_loc (CNO 質量分率) を配列で渡すと
    非一様組成で評価する。省略時はグローバル一様値 (X, Z/2)。
    """
    T6    = np.maximum(T, 1.0) / 1.0e6
    T9    = T6 / 1.0e3
    X_e   = X if X_loc is None else X_loc
    X_CNO = (Z / 2.0) if XCNO_loc is None else XCNO_loc
    good  = T6 >= 0.5

    eps_pp  = np.zeros_like(T, dtype=float)
    eps_cno = np.zeros_like(T, dtype=float)
    if np.any(good):
        T6g = T6[good]; T9g = T9[good]; rhog = rho[good]
        X_eg   = X_e[good]   if np.ndim(X_e)   else X_e
        X_CNOg = X_CNO[good] if np.ndim(X_CNO) else X_CNO
        g11  = 1.0 + 3.82*T9g + 1.51*T9g**2 + 0.144*T9g**3 - 0.0114*T9g**4
        g141 = 1.0 - 2.00*T9g + 3.41*T9g**2 - 2.43*T9g**3
        eps_pp[good]  = (PP_CALIB * 2.57e6 * g11
                         * rhog * X_eg**2 * T6g**(-2.0/3.0)
                         * np.exp(-33.81 * T6g**(-1.0/3.0)))
        eps_cno[good] = (CNO_CALIB * 8.24e27 * np.maximum(g141, 0.0)
                         * rhog * X_eg * X_CNOg * T6g**(-2.0/3.0)
                         * np.exp(-152.31 * T6g**(-1.0/3.0) - (T9g/0.8)**2))
    return eps_pp, eps_cno


def energy_generation(rho, T, X_loc=None, XCNO_loc=None):
    pp, cno = energy_generation_components(rho, T, X_loc, XCNO_loc)
    return pp + cno


# ══════════════════════════════════════════════════════
# 2.  不透明度 / nabla
# ══════════════════════════════════════════════════════

try:
    import stellar_structure_phase4 as _ph4
    _HAVE_PH4 = True
except ImportError:
    _HAVE_PH4 = False

# ══════════════════════════════════════════════════════
# 2a. 内蔵 OPAL96 テーブルリーダー (GN93hz.dat)
#     Phase 4 依存を持たず単体で動作する。
# ══════════════════════════════════════════════════════

def _parse_gn93hz(filepath):
    """GN93hz.dat を解析して OPAL96 テーブル群を返す (Phase 4 依存なし)。"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    tables = []
    i = 0; n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("TABLE #") or "X=" not in line:
            i += 1; continue
        meta = {}
        for tok in line.split():
            for key in ["X=", "Y=", "Z=", "dXc=", "dXo="]:
                if tok.startswith(key):
                    try: meta[key[:-1]] = float(tok.split("=")[1])
                    except Exception: pass
        if "X" not in meta: i += 1; continue
        j = i + 1; logR_arr = None
        while j < n and j < i + 12:
            l = lines[j].strip()
            if l.startswith("logT") and "-8" in l:
                logR_arr = np.array([float(x) for x in l.split()[1:]], dtype=float)
                j += 1; break
            j += 1
        if logR_arr is None: i += 1; continue
        logT_list = []; logK_rows = []
        while j < n:
            l = lines[j].strip()
            if not l: j += 1; continue
            if l.startswith("TABLE #"): break
            toks = l.split()
            if len(toks) >= 2:
                try:
                    logT = float(toks[0])
                    row = []
                    for v in toks[1:]:
                        row.append(np.nan if v == "9.999" else float(v))
                    while len(row) < len(logR_arr): row.append(np.nan)
                    logT_list.append(logT); logK_rows.append(row[:len(logR_arr)])
                except ValueError: pass
            j += 1
        if logT_list:
            tables.append({
                "X": meta["X"],
                "Y": meta.get("Y", 1.0 - meta["X"] - meta.get("Z", 0.0)),
                "Z": meta.get("Z", 0.0), "dXc": meta.get("dXc", 0.0),
                "logT": np.array(logT_list, dtype=float),
                "logR": logR_arr,
                "logK": np.array(logK_rows, dtype=float),
            })
        i = j
    return tables


class _OpalTable:
    """GN93hz.dat から kappa(rho, T) を返す最小クラス (Phase 4 依存なし)。"""
    def __init__(self, filepath, X_target, Z_target):
        from scipy.interpolate import RegularGridInterpolator
        all_tables = _parse_gn93hz(filepath)
        if not all_tables:
            raise RuntimeError(f"No OPAL tables found in {filepath}")
        Z_avail = sorted(set(t["Z"] for t in all_tables))
        Z_near = min(Z_avail, key=lambda z: abs(z - Z_target))
        tbls_Z = [t for t in all_tables
                  if abs(t["Z"] - Z_near) < 1e-5 and abs(t.get("dXc", 0.0)) < 1e-12]
        if not tbls_Z:
            raise RuntimeError(f"No OPAL tables for Z~{Z_near}")
        X_avail = sorted(set(t["X"] for t in tbls_Z))
        X_lo = max((x for x in X_avail if x <= X_target), default=X_avail[0])
        X_hi = min((x for x in X_avail if x >= X_target), default=X_avail[-1])
        tbl_lo = [t for t in tbls_Z if abs(t["X"] - X_lo) < 1e-5][0]
        tbl_hi = [t for t in tbls_Z if abs(t["X"] - X_hi) < 1e-5][0]
        logT = tbl_lo["logT"]; logR = tbl_lo["logR"]
        logK_lo = tbl_lo["logK"]; logK_hi = tbl_hi["logK"]
        w_hi = 0.0 if abs(X_hi - X_lo) < 1e-12 else (X_target - X_lo) / (X_hi - X_lo)
        logK = np.where(np.isnan(logK_lo) | np.isnan(logK_hi), np.nan,
                        (1-w_hi)*np.nan_to_num(logK_lo, nan=-99.) + w_hi*np.nan_to_num(logK_hi, nan=-99.))
        for iR in range(logK.shape[1]):
            col = logK[:, iR]; valid = ~np.isnan(col)
            if valid.sum() >= 2:
                idx = np.where(valid)[0]
                if idx[0] > 0: logK[:idx[0], iR] = col[idx[0]]
                if idx[-1] < len(col)-1: logK[idx[-1]+1:, iR] = col[idx[-1]]
                nm = np.isnan(logK[:, iR])
                if np.any(nm): logK[nm, iR] = np.interp(logT[nm], logT[~nm], logK[:, iR][~nm])
        self._logT = logT; self._logR = logR; self._logK = logK
        self._logT_min = float(logT[0]); self._logT_max = float(logT[-1])
        self._logR_min = float(logR[0]); self._logR_max = float(logR[-1])
        self._interp = RegularGridInterpolator(
            (logT, logR), logK, method="linear", bounds_error=False, fill_value=None)

    def kappa_array(self, rho, T):
        rho = np.maximum(np.asarray(rho, float), 1e-99)
        T   = np.maximum(np.asarray(T,   float), 1.0)
        logT = np.log10(T)
        logR = np.log10(rho / np.maximum((T/1e6)**3, 1e-99))
        logT_c = np.clip(logT, self._logT_min, self._logT_max)
        logR_c = np.clip(logR, self._logR_min, self._logR_max)
        pts = np.column_stack([np.ravel(logT_c), np.ravel(logR_c)])
        logK = self._interp(pts).reshape(np.shape(logT_c))
        return np.maximum(10.0**logK, 1e-4)


_OPAL_CACHE = {}   # キャッシュ: (path, X, Z) → _OpalTable


def _get_opal_table(filepath, X_target, Z_target):
    """OPAL テーブルをキャッシュ付きで取得する (Phase 4 依存なし)。"""
    key = (str(filepath), float(X_target), float(Z_target))
    if key not in _OPAL_CACHE:
        _OPAL_CACHE[key] = _OpalTable(filepath, float(X_target), float(Z_target))
    return _OPAL_CACHE[key]


# ══════════════════════════════════════════════════════
#  Phase D-1: 低温 opacity (Ferguson et al. 2005)
# ══════════════════════════════════════════════════════

def _parse_f05(filepath):
    """F05_lowT_g98.dat (g98.*.tron の連結ファイル) をパースする。
    各テーブル: 1行目に "... X= 0.700000 and Z= 0.020000"、
    "log T" で始まる logR ヘッダ行、以降 logT 降順のデータ行。
    Returns: list of dict(X, Z, logT[asc], logR, logK[nT,nR])"""
    import re as _re
    tables = []
    cur = None
    with open(filepath) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            m = _re.search(r'X=\s*([\d.eE+-]+)\s+and\s+Z=\s*([\d.eE+-]+)', line)
            if m:
                if cur and cur['rows']:
                    tables.append(cur)
                cur = dict(X=float(m.group(1)), Z=float(m.group(2)),
                           logR=None, rows=[])
                continue
            if cur is None:
                continue
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith('log t'):
                cur['logR'] = np.array([float(v) for v in s.split()[2:]])
                continue
            toks = s.split()
            try:
                vals = [float(v) for v in toks]
            except ValueError:
                continue
            if cur['logR'] is not None and len(vals) == len(cur['logR']) + 1:
                cur['rows'].append(vals)
    if cur and cur['rows']:
        tables.append(cur)
    out = []
    for t in tables:
        arr = np.array(t['rows'], dtype=float)
        logT = arr[:, 0]; logK = arr[:, 1:]
        idx = np.argsort(logT)                      # 降順 → 昇順
        out.append(dict(X=t['X'], Z=t['Z'], logT=logT[idx],
                        logR=t['logR'], logK=logK[idx]))
    return out


class _LowTTable:
    """Ferguson 2005 低温テーブルから κ(ρ,T) を返す (X 線形補間, Z 最近傍)。"""
    def __init__(self, filepath, X_target, Z_target):
        from scipy.interpolate import RegularGridInterpolator
        all_t = _parse_f05(filepath)
        if not all_t:
            raise RuntimeError(f"No F05 tables found in {filepath}")
        Z_avail = sorted(set(t['Z'] for t in all_t))
        Z_near = min(Z_avail, key=lambda z: abs(z - Z_target))
        tz = [t for t in all_t if abs(t['Z'] - Z_near) < 1e-12]
        X_avail = sorted(set(t['X'] for t in tz))
        X_lo = max((x for x in X_avail if x <= X_target), default=X_avail[0])
        X_hi = min((x for x in X_avail if x >= X_target), default=X_avail[-1])
        t_lo = [t for t in tz if abs(t['X'] - X_lo) < 1e-12][0]
        t_hi = [t for t in tz if abs(t['X'] - X_hi) < 1e-12][0]
        w = 0.0 if abs(X_hi - X_lo) < 1e-12 else (X_target - X_lo) / (X_hi - X_lo)
        logK = (1 - w) * t_lo['logK'] + w * t_hi['logK']
        self._logT = t_lo['logT']; self._logR = t_lo['logR']
        self._logT_min = float(self._logT[0]);  self._logT_max = float(self._logT[-1])
        self._logR_min = float(self._logR[0]);  self._logR_max = float(self._logR[-1])
        self._interp = RegularGridInterpolator(
            (self._logT, self._logR), logK,
            method="linear", bounds_error=False, fill_value=None)

    def logkappa(self, rho, T):
        rho = np.maximum(np.asarray(rho, float), 1e-99)
        T   = np.maximum(np.asarray(T,   float), 1.0)
        logT = np.clip(np.log10(T), self._logT_min, self._logT_max)
        logR = np.clip(np.log10(rho / np.maximum((T/1e6)**3, 1e-99)),
                       self._logR_min, self._logR_max)
        pts = np.column_stack([np.ravel(logT), np.ravel(logR)])
        return self._interp(pts).reshape(np.shape(logT))


_LOWT_CACHE = {}
_LOWT_WARNED = [False]
_OPAC_NOTICE = {'bad_path': False, 'source': None}


def _opac_notice_once(src):
    """使用中の opacity 源を一度だけ表示する (黙ったフォールバックの再発防止)。"""
    if _OPAC_NOTICE['source'] != src:
        print(f"[opacity] 使用中: {src}")
        _OPAC_NOTICE['source'] = src


def _get_lowT_table(args):
    """低温テーブルをキャッシュ付きで取得。見つからなければ None (従来動作)。"""
    path = getattr(args, 'lowT_table', None)
    if path is None:
        _sd = os.path.dirname(os.path.abspath(__file__))
        for c in [os.path.join(_sd, 'F05_lowT_g98.dat'), 'F05_lowT_g98.dat']:
            if os.path.exists(c):
                path = c
                break
    if path is None or not os.path.exists(path):
        if not _LOWT_WARNED[0]:
            print("[Phase D] 低温 opacity テーブル F05_lowT_g98.dat が見つかりません。"
                  " logT<3.75 は GN93hz 端の値でクリップされます (従来動作)。")
            _LOWT_WARNED[0] = True
        return None
    key = (str(path), float(getattr(args, 'opal_X', X)),
           float(getattr(args, 'opal_Z', Z)))
    if key not in _LOWT_CACHE:
        _LOWT_CACHE[key] = _LowTTable(path, key[1], key[2])
    return _LOWT_CACHE[key]


def opacity(rho, T, args, X_loc=None, Z_loc=None):
    """
    不透明度 κ [cm²/g]。

    Phase E: X_loc (H 質量分率, スカラー or 配列) を渡すと非一様組成で
    テーブルを X 補間する。省略時はグローバル一様 X (従来動作)。
    """
    # 非一様 X の判定 (配列で、かつ有意なばらつきがある場合のみ X 補間経路)
    _Xarr = None
    if X_loc is not None and np.ndim(X_loc) > 0 and np.size(X_loc) > 1:
        _Xarr = np.asarray(X_loc, float)
    _Xscal = (float(np.mean(X_loc)) if X_loc is not None else X)
    # ── 1. 内蔵 OPAL ──────────────────────────────────────────────
    # Phase D-0: args.opal_table が指すファイルが存在しない場合も自動探索に
    # フォールバックする (旧版はここで黙って解析式に落ちていた)。
    _opal_path = getattr(args, 'opal_table', None)
    if _opal_path is not None and not os.path.exists(str(_opal_path)):
        if not _OPAC_NOTICE['bad_path']:
            print(f"[opacity] 警告: opal_table={_opal_path} が存在しません。"
                  " 自動探索に切り替えます。")
            _OPAC_NOTICE['bad_path'] = True
        _opal_path = None
    if _opal_path is None:
        # スクリプトと同じディレクトリ → カレントディレクトリの順に探す
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        for _candidate in [os.path.join(_script_dir, 'GN93hz.dat'), 'GN93hz.dat']:
            if os.path.exists(_candidate):
                _opal_path = _candidate
                break
    if _opal_path and os.path.exists(_opal_path):
        try:
            _Zsel = getattr(args, 'opal_Z', Z) if Z_loc is None \
                else float(np.mean(Z_loc))

            def _opal_kappa(_rho, _T):
                """OPAL κ を返す。非一様 X なら X アンカー間を点別線形補間。"""
                if _Xarr is None:
                    return _get_opal_table(_opal_path, _Xscal, _Zsel
                                           ).kappa_array(_rho, _T)
                # X アンカー (OPAL ネイティブ格子に近い値) で挟んで補間
                anchors = np.array([0.0, 0.1, 0.35, 0.5, 0.7, 0.8])
                xlo, xhi = _Xarr.min(), _Xarr.max()
                sel = anchors[(anchors >= xlo) & (anchors <= xhi)]
                lo_a = anchors[anchors <= xlo]; hi_a = anchors[anchors >= xhi]
                use = np.unique(np.concatenate([
                    [lo_a.max()] if lo_a.size else [anchors[0]],
                    sel,
                    [hi_a.min()] if hi_a.size else [anchors[-1]]]))
                kap_a = np.array([_get_opal_table(_opal_path, float(xa), _Zsel)
                                  .kappa_array(_rho, _T) for xa in use])  # (nA, npts)
                logk_a = np.log10(np.maximum(kap_a, 1e-99))
                Xr = np.ravel(_Xarr)
                out = np.empty_like(Xr)
                for i in range(Xr.size):
                    out[i] = np.interp(Xr[i], use, logk_a[:, i])
                return 10.0 ** out.reshape(np.shape(_rho))

            kap_opal = _opal_kappa(rho, T)
            # ── Phase D-1: 低温側を Ferguson 2005 でブレンド ──
            blend  = getattr(args, 'lowT_blend_logT', 4.0)
            dlt    = max(getattr(args, 'lowT_blend_dlogT', 0.05), 1e-6)
            logT_v = np.log10(np.maximum(np.asarray(T, float), 1.0))
            if np.all(logT_v >= blend + dlt):
                _opac_notice_once("OPAL(GN93hz) + F05 低温ブレンド")
                return kap_opal          # 全点が高温側 → テーブル読込を省略
            _lt = _get_lowT_table(args)
            if _lt is None:
                return kap_opal          # フォールバック (従来動作)
            logk_low = _lt.logkappa(rho, T)   # 低温側は X 弱依存: 平均 X で十分
            s = np.clip((logT_v - (blend - dlt)) / (2.0 * dlt), 0.0, 1.0)
            w = s * s * (3.0 - 2.0 * s)  # smoothstep
            logk = w * np.log10(np.maximum(kap_opal, 1e-99)) + (1.0 - w) * logk_low
            _opac_notice_once("OPAL(GN93hz) + F05 低温ブレンド")
            return np.maximum(10.0 ** logk, 1e-6)
        except Exception:
            pass

    # ── 2. Phase 4 (内蔵 OPAL が使えない場合) ─────────────────────
    if _HAVE_PH4:
        try: return _ph4.opacity_from_args(rho, T, args)
        except Exception: pass

    # ── 3. フォールバック ────────────────────────────────────────
    _opac_notice_once("解析フォールバック (κ_es+Kramers+H⁻) — テーブル不使用!")
    T   = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    _Xf = _Xarr if _Xarr is not None else _Xscal
    _Zf = Z if Z_loc is None else Z_loc
    kap_es = getattr(args, 'kappa_es_factor', 1.0) * 0.2 * (1.0 + _Xf)
    # Kramers bf+ff: T > 10^6 K では急速に不正確になるので指数抑制
    kramers_w = np.exp(-np.maximum(np.maximum(T, 1.0) - 1e6, 0.0) / 5e5)
    kap_kr    = (getattr(args, 'kappa_kramers_factor', 1.0)
                 * 4.34e25 * _Zf * (1.0 + _Xf) * rho * np.maximum(T, 1.0)**(-3.5) * kramers_w)
    # H⁻ opacity (Hansen & Kawaler 標準形): 低温外層で重要
    #   κ_H⁻ = 2.5e-31 (Z/0.02) ρ^{1/2} T^9   [有効帯 3000 < T < 8000 K]
    # Phase A 修正: 旧式 7.5e-23 Z ρ^0.5 T^7.7 は太陽光球 (T=5777K, ρ~2e-7)
    # で κ≈62 cm²/g を返し正しい値 (~0.8) の約77倍過大だった。
    # 光球圧力 P_phot=2g/3κ が 1/50 になり表面対流層が異常に浅くなる原因。
    T_hm       = np.maximum(T, 1.0)
    taper_hot  = 1.0 / (1.0 + (T_hm / 8000.0)**12)
    taper_cool = 1.0 / (1.0 + (3000.0 / T_hm)**12)
    kap_hm     = (getattr(args, 'kappa_hminus_factor', 1.0)
                  * 2.5e-31 * (Z / 0.02) * np.maximum(rho, 1e-30)**0.5 * T_hm**9
                  * taper_hot * taper_cool)
    return np.maximum(kap_es + kap_kr + kap_hm, 1e-6)


def _kappa_hminus_scalar(rho_v, T_v, factor=1.0):
    """
    H⁻ 不透明度のスカラー版 (surface_bc_values 用)。
    opacity() 内のベクトル版と同一の式 (Hansen & Kawaler 形式 + テーパー)。
    """
    T_s = max(float(T_v), 1.0)
    taper_hot  = 1.0 / (1.0 + (T_s / 8000.0)**12)
    taper_cool = 1.0 / (1.0 + (3000.0 / T_s)**12)
    return (factor * 2.5e-31 * (Z / 0.02)
            * max(float(rho_v), 1e-30)**0.5 * T_s**9 * taper_hot * taper_cool)


def nabla_ad(rho, T, P, args, mu_loc=None):
    if _HAVE_PH4 and mu_loc is None:
        try: return _ph4.nabla_adiabatic_eff(rho, T, P, args)
        except Exception: pass
    mu_e = mu if mu_loc is None else mu_loc
    beta = rho * k_B * T / (mu_e * m_H) / np.maximum(P, 1e-99)
    return (8.0 - 6.0*beta) / (32.0 - 24.0*beta - 3.0*beta**2)


def nabla_rad(rho, T, P, L, m, kap, eps=None, center_points=3):
    """
    放射温度勾配 ∇_rad = 3κLP / (16π a c G m T⁴)   (K&W 5.28)

    Phase A 修正:
    1. 旧フォールバック式は分母が 64π で標準の 1/4 だったバグを修正。
    2. 中心近傍 (最初の center_points 点) では L/m が 0/0 の不定形になり、
       差分評価が ∇_rad の数値スパイク (偽対流コアの原因) を生むため、
       級数展開の極限  L/m → ε  (m→0)  を使って解析的に評価する:
         ∇_rad(0) = 3 κ_c ε_c P_c / (16π a c G T_c⁴)
    3. Phase 4 への委譲を廃止し内蔵実装に統一 (環境非依存)。
    """
    T_s = np.maximum(T, 1.0)
    Lom = np.maximum(L, 0.0) / np.maximum(m, 1e-99)
    Lom = np.atleast_1d(np.asarray(Lom, dtype=float))
    k_c = int(min(center_points, len(Lom) - 1))
    if eps is not None and k_c > 0:
        eps_arr = np.maximum(np.atleast_1d(np.asarray(eps, dtype=float)), 0.0)
        Lom = Lom.copy()
        Lom[:k_c] = eps_arr[:k_c]
    out = 3.0 * kap * P * Lom / (16.0 * np.pi * a_rad * c_light * G * T_s**4)
    return np.clip(out, 0.0, 1e4)


def _mlt_cubic_eta(U, W, n_bisect=64):
    """
    Phase D-2: K&W §7 完全 MLT 立方方程式の共通ソルバー (ベクトル化)。

      f(η) = η³ + (8U/9)(η² + 2Uη − W) = 0,   η = ξ − U,   W = ∇_rad − ∇_ad

    f(0) = −8UW/9 < 0, f(η_max) = η_max³ > 0 (η_max = √(U²+W)−U),
    f'(η) > 0 (η>0, U>0) より根は (0, η_max] に唯一 → 二分法が必ず収束。
    η_max は桁落ちしない形 W/(√(U²+W)+U) で評価する。
    64 回の二分で相対精度 ~2⁻⁶⁴ (倍精度の限界)。

    実効勾配は ∇ = ∇_ad + η² + 2Uη  (= ∇_ad + ξ² − U²)。
    効率極限 U→0: η→(8UW/9)^{1/3}→0 で ∇→∇_ad、
    非効率極限 U→∞: η→η_max で ∇→∇_rad を自動的に再現する。
    """
    U = np.atleast_1d(np.asarray(U, dtype=float))
    W = np.atleast_1d(np.asarray(W, dtype=float))
    W_pos = np.maximum(W, 0.0)
    hi = W_pos / (np.sqrt(U * U + W_pos) + np.maximum(U, 1e-300))
    lo = np.zeros_like(hi)
    c1 = 8.0 * U / 9.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        f = mid**3 + c1 * (mid * mid + 2.0 * U * mid - W_pos)
        neg = f < 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def mlt_nabla(nr, na, rho, T, P, r, m, L, kap, args):
    """
    混合長理論 (MLT) による実効温度勾配を返す。

    Phase D-2: 完全 MLT (K&W §7 立方方程式) を標準採用。
      δ = (4−3β)/β  (理想気体+放射),  c_P = Pδ/(TρΔ_ad)  (熱力学恒等式),
      U = (3acT³)/(c_P ρ² κ ℓ²) √(8H_P/(gδ)),  ℓ = α H_P
      → _mlt_cubic_eta で η を解き ∇ = ∇_ad + η² + 2Uη。
    旧・弱対流近似 (Henyey 1965) は args.mlt_solver='weak' で残置 (A/B 用)。

    戻り値: (nabla_eff, dummy×5)  ← Phase 4 と同じシグネチャ
    """
    alpha_mlt = getattr(args, 'mlt_alpha', 1.8)

    if getattr(args, 'mlt_solver', 'cubic') == 'cubic':
        g_loc = G * np.maximum(m, 1e-30) / np.maximum(r, 1e-10)**2
        H_P   = np.maximum(P, 1e-30) / (np.maximum(rho, 1e-30)
                                        * np.maximum(g_loc, 1e-30))
        beta  = np.clip(rho * k_B * T / (mu * m_H) / np.maximum(P, 1e-30),
                        1e-6, 1.0)
        delta = (4.0 - 3.0 * beta) / beta
        na_s  = np.clip(na, 1e-3, None)
        c_P   = P * delta / (np.maximum(T, 1.0) * np.maximum(rho, 1e-30) * na_s)
        ell_m = alpha_mlt * H_P
        U = (3.0 * a_rad * c_light * np.maximum(T, 1.0)**3
             / (c_P * np.maximum(rho, 1e-30)**2 * np.maximum(kap, 1e-10)
                * np.maximum(ell_m, 1e-10)**2)
             * np.sqrt(8.0 * H_P / (np.maximum(g_loc, 1e-30) * delta)))
        U = np.clip(U, 1e-30, 1e30)
        W = np.maximum(nr - na, 0.0)
        eta = _mlt_cubic_eta(U, W)
        nabla_eff = na + eta * (eta + 2.0 * U)
        nabla_eff = np.clip(nabla_eff, na, np.maximum(nr, na))
        dummy = np.zeros_like(np.atleast_1d(nr), dtype=float)
        return nabla_eff, dummy, dummy, dummy, dummy, dummy

    # ── 旧・弱対流近似 (mlt_solver='weak', 検証比較用) ────────────

    # 圧力スケールハイト H_P = P / (ρ g)
    g_loc  = G * np.maximum(m, 1e-30) / np.maximum(r, 1e-10)**2
    H_P    = np.maximum(P, 1e-30) / (np.maximum(rho, 1e-30) * np.maximum(g_loc, 1e-30))

    # 比熱 c_p (理想気体 + 放射): c_p = (γ/(γ-1)) × k_B/(mu mH)
    beta  = rho * k_B * T / (mu * m_H) / np.maximum(P, 1e-30)
    gamma = (32 - 24*beta - 3*beta**2) / (24 - 21*beta)  # 有効断熱指数
    c_p   = np.where(gamma > 1.0,
                     gamma / (gamma - 1.0) * k_B / (mu * m_H),
                     5.0 / 2.0 * k_B / (mu * m_H))

    # MLT パラメータ (Cox & Giuli, 無次元)
    A    = np.maximum(nr - na, 0.0)
    xi   = kap * np.maximum(rho, 1e-30)**2 * c_p * np.maximum(H_P, 1e-10)**2 * alpha_mlt**2
    denom = 6.0 * sigma_sb * np.maximum(T, 1.0)**3
    B    = np.where(denom > 0, xi / np.maximum(denom, 1e-99), 1e10)

    # 3次方程式の近似解 (弱対流: Henyey et al. 1965)
    W    = np.where(B > 0, 9.0 * A / np.maximum(B, 1e-30), 0.0)
    W    = np.minimum(W, 1e10)
    # ∇_eff ≈ ∇_ad + A / (1 + W/9)  (弱対流近似)
    # より正確: 3次方程式を解く
    # y³ + y² + y - W/27 = 0 の解 y → ∇_eff = ∇_ad + A(1 - y/(1+y+y²))
    # 簡略: y ≈ (W/27)^{1/3} for small W
    y    = np.where(W > 0, np.minimum((W / 27.0)**(1.0/3.0), A), 0.0)
    nabla_eff = na + A * (1.0 - y**2 / np.maximum(1.0 + y + y**2, 1e-30))
    nabla_eff = np.clip(nabla_eff, na, nr)

    dummy = np.zeros_like(nr)
    return nabla_eff, dummy, dummy, dummy, dummy, dummy


def choose_nabla(q, S, p, ell, t, log_L_est, args):
    """
    ell = L/L_est から物理量を復元して nabla を計算する。

    【Opus の指摘に基づく平滑化】
    旧実装の np.where(unstable, nmlt, nr) はハードスイッチで、
    対流境界付近で nabla に不連続なキンクを生じさせる。
    このキンクが |res|∞ ≈ 3–4e-2 の収束壁の原因。

    修正: disc = nr − na を判別量として tanh でブレンド。
      w = 0.5 × (1 + tanh(disc / eps_schwarz))
      nabla = nr + (nmlt − nr) × w
    disc < 0 (放射安定) → w → 0 → nabla → nr
    disc > 0 (対流不安定) → w → 1 → nabla → nmlt
    境界付近は滑らかに補間されるので残差が C∞ になる。

    キャップも soft cap (tanh) に変更して二次的なキンクを除去。
    """
    r   = R_sun * np.exp(np.clip(S, -80, 20))
    L_p = L_sun * np.exp(float(log_L_est)) * np.clip(ell, 0, None)
    m_p = q * M_hat * M_sun
    P_p = P_REF * np.exp(np.clip(p,  -100, 20))
    T_p = T_REF * np.exp(np.clip(t,  -80,  10))
    X_q, Y_q, Z_q, mu_q, XCNO_q = _comp_at(q)   # Phase E: 非一様組成
    rho = density_from_PT(P_p, T_p, mu_loc=mu_q)

    kap = opacity(rho, T_p, args, X_loc=X_q, Z_loc=Z_q)
    eps = energy_generation(rho, T_p, X_loc=X_q, XCNO_loc=XCNO_q)
    nr  = nabla_rad(rho, T_p, P_p, L_p, m_p, kap, eps=eps)
    na  = nabla_ad(rho, T_p, P_p, args, mu_loc=mu_q)

    nmlt, *_ = mlt_nabla(nr, na, rho, T_p, P_p, r, m_p, L_p, kap, args)

    # ── 平滑化ブレンド (tanh) ──────────────────────────────
    eps_schwarz = getattr(args, 'schwarz_smooth_eps', 0.03)
    disc = nr - na
    w = 0.5 * (1.0 + np.tanh(disc / max(eps_schwarz, 1e-12)))
    nabla = nr + (nmlt - nr) * w      # 放射→対流を滑らかに補間

    # ── ソフトキャップ (中心安定用) ──────────────────────────
    # 旧: np.clip(nabla, 0, cap)  → キンクを作る
    # 新: cap × tanh(nabla / cap) → 滑らかな上限
    q_nc = getattr(args, 'q_nabla_center', 0.002)
    q_nw = max(getattr(args, 'q_nabla_width', 0.002), 1e-12)
    cw   = 1.0 - sigmoid((q - q_nc) / q_nw)
    cap  = getattr(args, 'nabla_hard_cap', 0.4) * (1-cw) + \
           getattr(args, 'nabla_center_cap', 0.4) * cw
    # 下限は 0 (タンジェント双曲線は原点で 0)
    nabla = np.maximum(nabla, 0.0)
    nabla = cap * np.tanh(nabla / np.maximum(cap, 1e-12))

    return nabla, nr, na


# ══════════════════════════════════════════════════════
# 3.  Pack / Unpack (4N+1 変数)
# ══════════════════════════════════════════════════════
# 増加列 (S, ell): 先頭値 + log-increments
# 減少列 (p, t):   log-drops + 表面値
# スカラー: log_L_est

def pack6(S, ell, p, t, log_L_est):
    """(S, ell, p, t: 各 N 要素, log_L_est: スカラー) → 長さ 4N+1 のベクトル。"""
    N = len(S)

    xs   = np.empty(N); xs[0]   = S[0];   xs[1:]   = np.log(np.maximum(np.diff(S),   1e-34))
    xell = np.empty(N); xell[0] = ell[0]; xell[1:] = np.log(np.maximum(np.diff(ell), 1e-34))

    xp = np.empty(N)
    xp[:-1] = np.log(np.maximum(p[:-1] - p[1:], 1e-34))
    xp[-1]  = p[-1]

    xt = np.empty(N)
    xt[:-1] = np.log(np.maximum(t[:-1] - t[1:], 1e-34))
    xt[-1]  = t[-1]

    return np.concatenate([xs, xell, xp, xt, [float(log_L_est)]])


def unpack6(X, N):
    """長さ 4N+1 → (S, ell, p, t: 各 N, log_L_est: スカラー)。"""
    xs   = X[0:N]
    xell = X[N:2*N]
    xp   = X[2*N:3*N]
    xt   = X[3*N:4*N]
    log_L_est = float(X[4*N])

    # S, ell: 増加列
    S   = np.empty(N); S[0]   = xs[0]
    ell = np.empty(N); ell[0] = xell[0]
    S[1:]   = S[0]   + np.cumsum(np.exp(np.clip(xs[1:],   -50, 50)))
    ell[1:] = ell[0] + np.cumsum(np.exp(np.clip(xell[1:], -50, 50)))

    # p, t: 減少列
    p_surf = xp[-1]
    p = np.empty(N); p[-1] = p_surf
    p[:-1] = p_surf + np.cumsum(np.exp(np.clip(xp[:-1], -80, 80))[::-1])[::-1]

    t_surf = xt[-1]
    t = np.empty(N); t[-1] = t_surf
    t[:-1] = t_surf + np.cumsum(np.exp(np.clip(xt[:-1], -80, 80))[::-1])[::-1]

    return S, ell, p, t, log_L_est


def physical6(q, S, p, ell, t, log_L_est):
    """無次元変数 → 物理量。Phase E: 非一様 μ(q) に対応。"""
    r   = R_sun * np.exp(np.clip(S,   -80, 20))
    m   = q * M_hat * M_sun
    P   = P_REF * np.exp(np.clip(p,  -100, 20))
    L   = L_sun * np.exp(float(log_L_est)) * np.clip(ell, 0, None)
    T   = T_REF * np.exp(np.clip(t,   -80, 10))
    _, _, _, mu_q, _ = _comp_at(q)
    rho = density_from_PT(P, T, mu_loc=mu_q)
    return r, m, P, L, T, rho


# ══════════════════════════════════════════════════════
# 4.  表面 BC
# ══════════════════════════════════════════════════════

def surface_bc_values(S_last, ell_last, log_L_est, M_hat_val, args):
    """
    表面境界条件の t_surf, p_surf を計算する (自己無撞着)。

    R_star = R_sun × exp(S_last)
    L_star = L_sun × exp(log_L_est) × ell_last
    T_eff  = (L_star / 4π R_star² σ)^{1/4}

    光球圧力 P_phot は Eddington の τ=2/3 近似:
      P_phot = (2/3) g / κ_phot

    κ_phot は T_eff での不透明度を使う:
      高温 (T_eff > 10^4 K): 電子散乱 κ_es のみ
      低温 (T_eff < 10^4 K): H⁻ + κ_es を使う
        κ_H⁻(T_eff, ρ_guess) の近似:
        ρ_guess ≈ P_phot_es × mu m_H / (k_B T_eff)  (電子散乱での初期推定)
    """
    R_star  = R_sun * np.exp(float(S_last))
    L_star  = L_sun * np.exp(float(log_L_est)) * max(float(ell_last), 1e-30)
    M_star  = M_hat_val * M_sun

    T_eff_v = max((L_star / (4*np.pi * R_star**2 * sigma_sb))**0.25, 1.0)
    g_eff   = G * M_star / R_star**2

    # 第1近似: 電子散乱のみで P_phot を推定
    kap_es  = getattr(args, 'kappa_es_factor', 1.0) * 0.2 * (1.0 + X)
    P_phot_es = (2.0/3.0) * g_eff / max(kap_es, 1e-10)

    # 低温星 (T_eff < 1.2×10^4 K) はテーブル opacity で κ_phot を反復決定
    # Phase D-1: 解析 H⁻ 近似 → opacity() (OPAL + F05 ブレンド) に統一。
    # 低温テーブル不在時も opacity() 内のフォールバック (κ_es+Kramers+H⁻)
    # が同等の値を返すので挙動は連続。
    if T_eff_v < 1.2e4:
        rho_phot_guess = max(P_phot_es * mu * m_H / (k_B * max(T_eff_v, 1.0)), 1e-12)
        kap_phot = float(opacity(np.array([rho_phot_guess]),
                                 np.array([T_eff_v]), args)[0])
        for _ in range(3):
            P_phot_iter = (2.0/3.0) * g_eff / max(kap_phot, 1e-10)
            rho_phot_guess = max(P_phot_iter * mu * m_H
                                 / (k_B * max(T_eff_v, 1.0)), 1e-12)
            kap_phot = float(opacity(np.array([rho_phot_guess]),
                                     np.array([T_eff_v]), args)[0])
        P_phot = (2.0/3.0) * g_eff / max(kap_phot, 1e-10)
    else:
        # 高温: 電子散乱のみで十分
        P_phot = P_phot_es

    t_surf  = np.log(T_eff_v / T_REF)
    p_surf  = np.log(max(P_phot, 1.0) / P_REF)
    return t_surf, p_surf, T_eff_v, P_phot, R_star, L_star


# ══════════════════════════════════════════════════════
# 5.  残差ベクトル
# ══════════════════════════════════════════════════════

def residual_vector6(X, q, args):
    """
    Phase 6 残差 (4N-1 個)。
    """
    N = len(q)
    S, ell, p, t, log_L_est = unpack6(X, N)

    # 表面 BC 値を S[-1], ell[-1], log_L_est から計算
    t_surf, p_surf, T_eff_val, P_phot, R_star, L_star = \
        surface_bc_values(S[-1], ell[-1], log_L_est, M_hat, args)

    dq    = np.diff(q)
    qmid  = 0.5*(q[:-1]+q[1:])
    Smid  = 0.5*(S[:-1]+S[1:])
    emid  = 0.5*(ell[:-1]+ell[1:])
    pmid  = 0.5*(p[:-1]+p[1:])
    tmid  = 0.5*(t[:-1]+t[1:])

    r, m, P, L, T, rho = physical6(qmid, Smid, pmid, emid, tmid, log_L_est)
    Xmid, Ymid, Zmid, mumid, XCNOmid = _comp_at(qmid)   # Phase E
    eps   = energy_generation(rho, T, X_loc=Xmid, XCNO_loc=XCNOmid)
    nabla, nr, na = choose_nabla(qmid, Smid, pmid, emid, tmid, log_L_est, args)

    # ノード上の ε (Rell と R_energy で共通使用)。
    # Rell を中点 ε、R_energy をノード ε で評価すると離散化不整合が生じ、
    # 「中心1ノードの温度スパイクが ∫εdm=L を満たしつつ Rell からは
    # 見えない」ε 点源の抜け道が生まれる。両者をノード台形積分に統一する。
    Xn_c, Yn_c, Zn_c, mun_c, XCNOn_c = _comp_at(q)       # ノード上の組成
    P_n   = P_REF * np.exp(np.clip(p, -200, 200))
    T_n   = T_REF * np.exp(np.clip(t, -100, 50))
    rho_n = density_from_PT(P_n, T_n, mu_loc=mun_c)
    eps_n = energy_generation(rho_n, T_n, X_loc=Xn_c, XCNO_loc=XCNOn_c)
    eps_tz = 0.5 * (eps_n[:-1] + eps_n[1:])   # 区間台形平均

    # ── Rs: 半径方程式 ───────────────────────────────────
    Rs_model = dq * M_hat * RHO_S / (rho * np.exp(3*np.clip(Smid, -80, 20)))
    Rs_act   = S[1:] - S[:-1]
    Rs_raw   = Rs_act - Rs_model
    # hybrid スケーリング
    floor_rs = getattr(args, 'rs_floor', 1e-8)
    w_rel    = getattr(args, 'rs_relative_weight', 1.0)
    w_abs    = getattr(args, 'outer_rs_raw_weight', 1.0)
    sw       = sigmoid((qmid - getattr(args, 'outer_q_start', 0.5)) /
                        max(getattr(args, 'outer_q_width', 0.1), 1e-12))
    denom    = np.abs(Rs_act) + np.abs(Rs_model) + floor_rs
    Rs = (1-sw) * w_rel * Rs_raw / denom + sw * w_abs * Rs_raw

    # ── Rp: 圧力方程式 ───────────────────────────────────
    Rp_model = -dq * M_hat**2 * qmid / np.exp(np.clip(pmid + 4*Smid, -200, 200))
    Rp = (p[1:]-p[:-1]) - Rp_model

    # ── Rell: 光度方程式 (eps_scale なし!) ──────────────
    # L = ell × L_est, dL/dm = eps
    # dell/dq = M_hat × eps / (EPS_REF × exp(log_L_est))
    # ell を分母に入れるのは誤り (L_est は定数なので)
    Rell_model = dq * M_hat * eps_tz / (EPS_REF * np.exp(float(log_L_est)))
    Rell = (ell[1:]-ell[:-1]) - Rell_model

    # ── Rt: 温度方程式 ────────────────────────────────────
    Rt = (t[1:]-t[:-1]) - nabla * Rp_model
    # 最表面の大気的区間 (1-q < outer_atm_dm, τ ≲ 2/3 相当) は
    # 灰色大気の T(τ) 構造を内部の ∇ 方程式で表現できず O(0.1-0.3) の
    # 残差床を作る。この床が内部収束の優先度を下げるため減重する。
    # (恒久対策は Phase C の大気/エンベロープ積分)
    _w_atm  = getattr(args, 'outer_rt_weight', 0.15)
    _dm_atm = getattr(args, 'outer_atm_dm', 3e-10)
    if _w_atm < 1.0:
        _atm = (1.0 - qmid) < _dm_atm
        Rt = np.where(_atm, _w_atm * Rt, Rt)

    # ── Rbc: 表面境界条件 ─────────────────────────────────
    Rbc_t   = t[-1]   - t_surf              # t[-1] = T_eff(L_star, R_star) の対数
    Rbc_p   = p[-1]   - p_surf              # p[-1] = P_phot(R_star) の対数
    Rbc_ell = ell[-1] - 1.0                 # ell[-1] = 1  が log_L_est を拘束

    # ── Rcen: 中心境界条件 ────────────────────────────────
    _r0, _m0, P0, _L0, T0, rho0 = physical6(
        np.array([q[0]]), np.array([S[0]]),
        np.array([p[0]]), np.array([ell[0]]), np.array([t[0]]), log_L_est)
    rho_c = float(rho0[0]); T_c = float(T0[0])
    _Xc, _Yc, _Zc, _muc, _XCNOc = _comp_at(np.array([q[0]]))
    eps_c = float(energy_generation(np.array([rho_c]), np.array([T_c]),
                                    X_loc=_Xc, XCNO_loc=_XCNOc)[0])

    # S[0]: r(q0) = (3 M_star q0 / 4π ρ_c)^{1/3} → S0 = log(r0/R_sun)
    S0_exp = (1.0/3.0) * np.log(max(3 * M_hat * q[0] / (4*np.pi * rho_c * RHO_S), 1e-200))
    # ell[0]: L(q0) = eps_c × M_star × q0 → ell0 = L(q0)/L_est
    L_q0   = eps_c * M_hat * M_sun * q[0]
    L_est  = L_sun * np.exp(float(log_L_est))
    ell0_exp = L_q0 / max(L_est, 1e-99)

    Rcen_S   = S[0]   - S0_exp
    Rcen_ell = ell[0] - ell0_exp

    # 中心の T と rho に対する弱拘束 (weight がゼロならゼロ残差)
    w_cT  = getattr(args, 'weight_center_T',   0.0)
    w_crho= getattr(args, 'weight_center_rho', 0.0)
    rho_unit_tgt = 150.0 * M_hat / max(np.exp(3*S[-1]), 1e-99)
    Rcen_T   = w_cT   * (t[0] - np.log(max(T_c/T_REF, 1e-99)))
    Rcen_rho = w_crho * np.log(max(rho_c / max(rho_unit_tgt, 1e-99), 1e-99))

    # M-L 事前情報: ZAMS (零年齢主系列) の正しい質量-光度関係
    # ZAMS での太陽 (M=1) は L ≈ 0.72 L☉ (現在の太陽の 72%)
    # 一般式: L_ZAMS ≈ L_ZAMS_solar × M^4   (近似)
    # L_ZAMS_solar = 0.72 → log_offset = log(0.72) = -0.3285
    #
    # w_ml_prior は収束が進むにつれて段階的に弱めることができる。
    # 具体的には args.w_ml_prior_scale (0〜1) を掛けて段階的に緩める。
    # 完全収束後は 0 にして純粋な物理解を得る。
    w_ml = getattr(args, 'w_ml_prior', 0.0) * getattr(args, 'w_ml_prior_scale', 1.0)
    if w_ml > 0:
        # ZAMS 光度スケール (M=1 で L=0.72 L☉)
        L_ZAMS_solar = getattr(args, 'L_zams_solar', 0.72)
        log_offset = np.log(max(L_ZAMS_solar, 1e-10))
        if M_hat >= 0.7:
            log_L_exp = 4.0 * np.log(max(M_hat, 0.1)) + log_offset
        else:
            log_L_exp = 4.5 * np.log(max(M_hat, 0.1)) + log_offset
        Rcen_ML = w_ml * (log_L_est - log_L_exp)
    else:
        Rcen_ML = 0.0

    # R_star 事前情報: Hayashi トラック偽解（R~3+ R_sun）への逃げを防ぐ。
    # ZAMS の半径スケール R ~ M^0.8 × R_zams を中心に soft penalty を課す。
    # tanh を使うことで遠い偽解（R=10 R_sun 超）でも残差が有界になる。
    # w_R_prior > 0 のとき有効。
    w_R = getattr(args, 'w_R_prior', 2.0)
    if w_R > 0.0:
        R_zams_s   = getattr(args, 'R_zams_solar', 0.87)
        S_target   = np.log(max(M_hat, 0.1) ** 0.8 * R_zams_s)
        _delta_R   = 0.5   # tanh の幅 (0.5 ≈ R が 1.6 倍以内で線形)
        Rcen_R     = w_R * np.tanh((float(S[-1]) - S_target) / _delta_R)
    else:
        Rcen_R = 0.0

    w_s = getattr(args, 'weight_structure', 1.0)
    w_b = getattr(args, 'weight_bc', 1.0)

    # ── Phase A-4: 大域エネルギー整合 (熱平衡 ZAMS の定義) ──────────
    # ∫ε dm = L_star を大域積分量として直接要求する。
    # Rell は区間ごとの式なので、各区間に O(0.01) の残差を薄く分散させる
    # ことで「核燃焼なしで L が流れる」偽解 (L_mass/L_star ≈ 0) が
    # 二乗和コスト的に安価に成立してしまう。この大域残差 1 本が
    # その偽解の谷を w_E 分だけ持ち上げる。真の ZAMS 解では厳密に 0。
    # log 比を線形評価 (±3 でクリップ)。tanh だと大きくずれた偽解で
    # 勾配が消失し引き戻せなくなるため、広い範囲で勾配を保つ。
    # 注意: ε は必ずノード値の台形積分で評価する。中点 ε だと、中心
    # ノード 1 点の温度スパイク (残差の中点サンプリングから見えない) で
    # 診断上の L_mass だけが暴走する離散化の抜け道が生じる。
    w_E = getattr(args, 'w_energy_balance', 2.0)
    if w_E > 0:
        L_mass_v = float(np.sum(eps_tz * dq)) * M_hat * M_sun
        L_star_v = L_sun * np.exp(float(log_L_est)) * max(float(ell[-1]), 1e-30)
        R_energy = w_E * float(np.clip(
            np.log(max(L_mass_v, 1e-30) / max(L_star_v, 1e-30)) / 2.0, -3.0, 3.0))
    else:
        R_energy = 0.0

    return np.concatenate([
        w_s * Rs,
        w_s * Rp,
        w_s * Rell,
        w_s * Rt,
        [w_b * Rbc_t, w_b * Rbc_p, w_b * Rbc_ell],
        [Rcen_S, Rcen_ell, Rcen_T, Rcen_rho, Rcen_ML, Rcen_R, R_energy],
    ])


# ══════════════════════════════════════════════════════
# 6.  初期モデル
# ══════════════════════════════════════════════════════

def solve_lane_emden(n=3.0, n_grid=4000):
    if _HAVE_PH4:
        return _ph4.solve_lane_emden(n, n_grid)
    xi = np.linspace(0, 10, n_grid); dxi = xi[1]-xi[0]
    theta = np.ones(n_grid); phi = np.zeros(n_grid)
    theta[1] = 1 - dxi**2/6; phi[1] = -dxi/3
    for i in range(1, n_grid-1):
        if theta[i] <= 0: break
        phi[i+1] = phi[i] + dxi*(-2/xi[i]*phi[i] - theta[i]**n)
        theta[i+1] = theta[i] + dxi*phi[i+1]
    idx1 = np.argmax(theta <= 0)
    return xi, theta, phi, xi[idx1], phi[idx1]


def make_q_mesh(N=80, q0=1e-8, power=1.4, mesh_style='composite',
                n_outer=40, outer_q_join=0.90, outer_dm_min=1e-11):
    """
    質量メッシュ q ∈ [q0, 1] を生成する。

    mesh_style='composite' (Phase B, デフォルト):
      内部 (q < outer_q_join): 従来の power 分布 N 点
      外層 (q > outer_q_join): ln(1-q) 均等 n_outer 点 + 表面点 q=1
      太陽型星の表面対流層 (q>0.976, 全質量の2.4%) は従来の均等系
      メッシュでは 2 点しか持てず構造的に解像できなかった。
      ln(1-q) 均等化により対流層に ~35 点、光球直下 (1-q~1e-11,
      Δτ≲1) まで到達し、表面区間の差分方程式の stiffness
      (旧: 1区間で Δln P≈20) を各区間 ~0.5 に分散する。

    mesh_style='legacy': 従来の分布 (再現性確認用)。
    """
    if mesh_style == 'legacy':
        if _HAVE_PH4 and N >= 40:
            return _ph4.make_q_mesh(N, q0, power)
        return q0 + (1-q0)*np.linspace(0,1,N)**power
    # composite
    base = q0 + (1-q0)*np.linspace(0,1,N)**power
    q_in = base[base < outer_q_join]
    x    = np.linspace(np.log(1.0-outer_q_join), np.log(outer_dm_min), int(n_outer))
    q_out = 1.0 - np.exp(x)
    # 中心対数細分: r ∝ q^{1/3} のため q[0]=1e-8 → q[1]~5e-3 の第1区間は
    # 対数半径で ΔS≈4.4 に及ぶ stiff 区間になる。log-q 均等 8 点で
    # ΔS≈0.5/区間に分散する (表面 ln(1-q) 細分の中心版)。
    q_cen = np.exp(np.linspace(np.log(max(q0, 1e-12)), np.log(1e-2), 8))
    q_all = np.unique(np.concatenate([q_cen, q_in, q_out, [1.0]]))
    # 浮動小数点起源の近接重複点 (dq→0 で残差を汚染) を相対間隔で除去
    keep = np.concatenate([[True],
                           np.diff(q_all)/np.maximum(q_all[1:], 1e-30) > 1e-6])
    keep[-1] = True
    return q_all[keep]


def make_initial_model6_envelope(q, M_hat_val, args,
                                 R_hat=None, L_hat=None):
    """
    Phase A/B 初期モデル: n=3 ポリトロープ内部 + 表面からの離散逆積分外層。

    従来の Lane-Emden 単純初期値は表面区間の差分方程式と大きく不整合で
    初期残差が O(10^5) に達し、ソルバーが Hayashi 偽解へ発散していた。
    本関数は:
      1. n=3 ポリトロープを ZAMS スケール (R = R_hat R☉) で構築 (内部)
      2. 表面 BC (surface_bc_values) から q_join=0.90 まで、離散化された
         構造方程式そのもの (residual_vector6 と同じ中点法) を各区間で
         Brent 法により解いて逆積分 (外層)
      3. 内側ポリトロープを接続点でオフセット接続
    温度勾配は局所 Schwarzschild ∇=min(∇_rad, ∇_ad) (効率対流近似)。
    複合メッシュ (make_q_mesh mesh_style='composite') と併用した場合、
    初期 |res|∞ は O(10) まで下がる。

    Returns: S0, ell0, p0, t0, log_Le0
    """
    from scipy.optimize import brentq as _brentq
    from scipy.integrate import solve_ivp as _solve_ivp
    global M_hat
    M_hat = float(M_hat_val)
    N = len(q)
    if R_hat is None:
        R_hat = getattr(args, 'R_zams_solar', 0.87) * max(M_hat, 0.1)**0.8
    if L_hat is None:
        L_z = getattr(args, 'L_zams_solar', 0.72)
        L_hat = L_z * max(M_hat, 0.1)**(4.0 if M_hat >= 0.7 else 4.5)

    # ── n=3 ポリトロープ (ZAMS スケール, 自己完結の積分) ──
    # 既存 solve_lane_emden() とは戻り値規約が異なるため自前で積分する。
    _s = _solve_ivp(
        lambda x, y: [y[1], -(max(y[0], 0.0))**3 - (2.0/max(x, 1e-9))*y[1]],
        [1e-6, 8.0], [1.0, 0.0], dense_output=True,
        rtol=1e-10, atol=1e-12, max_step=0.01)
    _xi_g = np.linspace(1e-6, 8.0, 4000)
    theta = _s.sol(_xi_g)[0]; _dth = _s.sol(_xi_g)[1]
    _i1 = int(np.argmax(theta <= 0.0))
    xi   = _xi_g[:_i1]; _dth = _dth[:_i1]
    theta = np.maximum(theta[:_i1], 1e-12)
    xi1  = float(xi[-1])
    alpha  = R_hat * R_sun / xi1
    rho_c0 = M_hat * M_sun / (4*np.pi * alpha**3 * max(-xi[-1]**2*_dth[-1], 1e-30))
    P_c0   = np.pi * G * alpha**2 * rho_c0**2
    T_c0   = P_c0 * mu * m_H / (rho_c0 * k_B)
    m_xi   = 4*np.pi*alpha**3*rho_c0*np.maximum(-xi**2*_dth, 0.0)
    q_xi   = np.clip(m_xi / max(m_xi[-1], 1e-30), 0, 1)
    r_pl   = np.interp(q, q_xi, alpha*np.asarray(xi))
    T_pl   = np.maximum(np.interp(q, q_xi, T_c0*theta),   3000.0)
    P_pl   = np.maximum(np.interp(q, q_xi, P_c0*theta**4), 1.0)
    S_pl = np.log(np.maximum(r_pl/R_sun, 1e-8))
    p_pl = np.log(P_pl/P_REF); t_pl = np.log(T_pl/T_REF)

    # ── ell プロファイル (ポリトロープ上の ε 積分) ──
    rho_pl = density_from_PT(P_pl, T_pl)
    eps_pl = energy_generation(rho_pl, T_pl)
    dqa = np.diff(q); Lc = np.zeros(N)
    for i in range(1, N):
        Lc[i] = Lc[i-1] + 0.5*(eps_pl[i-1]+eps_pl[i])*M_hat*M_sun*dqa[i-1]
    ell0 = np.maximum(Lc/max(Lc[-1], 1e-99), 1e-14); ell0[-1] = 1.0
    log_Le0 = float(np.log(max(L_hat, 1e-10)))
    L_est = L_sun * np.exp(log_Le0)

    # ── 表面 BC → q_join まで離散逆積分 ──
    k_join = int(np.argmin(np.abs(q - 0.90)))
    S = S_pl.copy(); p = p_pl.copy(); t = t_pl.copy()
    S[-1] = np.log(max(R_hat, 1e-6))
    ts, ps, _, _, _, _ = surface_bc_values(S[-1], 1.0, log_Le0, M_hat, args)
    t[-1] = ts; p[-1] = ps
    # 外層の t は「光球 (p_surf, t_surf) ↔ 接続点ポリトロープ (p_pl, t_pl)」
    # を ln P で線形ブリッジする。∇=min(∇_rad,∇_ad) による逆積分は
    # 超断熱層 (Phase C の領域) を表現できず対流層エントロピーが決まらない
    # ため断熱線が数10 MK まで暴走する。ブリッジは両端固定で頑健。
    _grad_eff = (t_pl[k_join] - t[-1]) / max(p_pl[k_join] - p[-1], 1e-6)
    for _pass in range(2):
        for k in range(N-2, k_join-1, -1):
            dq = q[k+1]-q[k]; qm = 0.5*(q[k]+q[k+1])
            if _pass == 0:
                p[k] = p[k+1]
            for _ in range(20):
                def _f_p(pk):
                    pm = 0.5*(pk+p[k+1]); Sm = 0.5*(S[k]+S[k+1])
                    return (p[k+1]-pk) + dq*qm*M_hat**2/np.exp(np.clip(pm+4*Sm, -200, 200))
                try: p_new = _brentq(_f_p, p[k+1], p[k+1]+30.0, xtol=1e-13)
                except ValueError: p_new = p[k]
                t_new = t[-1] + _grad_eff*(p_new - p[-1])
                pm = 0.5*(p_new+p[k+1]); tm = 0.5*(t_new+t[k+1])
                P_m = P_REF*np.exp(np.clip(pm, -200, 200))
                T_m = T_REF*np.exp(np.clip(tm, -100, 50))
                rho_m = float(density_from_PT(np.array([P_m]), np.array([T_m]))[0])
                _Sm2 = 0.5*(S[k]+S[k+1])
                _Rs_m = dq*M_hat*RHO_S/(max(rho_m,1e-30)*np.exp(3*np.clip(_Sm2, -80, 20)))
                S_new = S[k+1] - min(_Rs_m, 1.0)
                dmax = max(abs(p_new-p[k]), abs(t_new-t[k]), abs(S_new-S[k]))
                p[k], t[k], S[k] = p_new, t_new, S_new
                if dmax < 1e-12: break

    # ── 内側ポリトロープを接続点でオフセット接続 ──
    S[:k_join] = S_pl[:k_join] + (S[k_join]-S_pl[k_join])
    p[:k_join] = p_pl[:k_join] + (p[k_join]-p_pl[k_join])
    t[:k_join] = t_pl[:k_join] + (t[k_join]-t_pl[k_join])
    S = np.maximum.accumulate(S)
    for arr in (p, t):
        for k in range(N-2, -1, -1):
            if arr[k] < arr[k+1] + 1e-12: arr[k] = arr[k+1] + 1e-12
    return S, ell0, p, t, log_Le0


def make_initial_model6(q_mesh, M_hat_val, args,
                        R_star_guess=None, L_star_guess=None):
    """Phase 6 初期モデル (Lane-Emden + 光球外挿)。"""
    global M_hat
    M_hat = float(M_hat_val)
    M_star = M_hat * M_sun

    xi, theta, phi, xi1, phi1 = solve_lane_emden()

    if R_star_guess is None: R_star_guess = R_sun * M_hat**0.8
    if L_star_guess is None:
        if M_hat <= 0.5: L_star_guess = L_sun * M_hat**5.0
        elif M_hat <= 2: L_star_guess = L_sun * M_hat**4.0
        else:            L_star_guess = L_sun * M_hat**3.5

    alpha  = R_star_guess / xi1
    rho_c  = M_star / (4*np.pi * alpha**3 * (-xi1**2 * phi1))

    r_le   = alpha * xi
    rho_le = rho_c * np.clip(theta, 0, None)**3
    T_c0   = T_REF * 1.1
    T_le   = np.maximum(T_c0 * np.clip(theta, 0, None), 100.0)
    P_le   = pressure_total(rho_le, T_le)
    m_le   = cumulative_trapezoid(4*np.pi*r_le**2*rho_le, r_le, initial=0.0)
    q_le   = m_le / M_star
    eps_le = energy_generation(rho_le, T_le)
    L_raw  = cumulative_trapezoid(4*np.pi*r_le**2*rho_le*eps_le, r_le, initial=0.0)

    q_le2, idx = np.unique(np.maximum.accumulate(q_le), return_index=True)
    r2=r_le[idx]; P2=P_le[idx]; T2=T_le[idx]; L2=L_raw[idx]
    q_le2[0] = 0.0

    q_ev = np.clip(q_mesh, q_le2[1], 1.0)
    r_i  = np.interp(q_ev, q_le2, r2)
    P_i  = np.interp(q_ev, q_le2, P2)
    T_i  = np.interp(q_ev, q_le2, T2)
    L_i  = np.interp(q_ev, q_le2, L2)

    # L_est 初期推定:
    # L_raw[-1] = ∫eps dm を使う。これにより ell = L_raw/L_raw[-1] が
    # 自然に [0,1] の単調増加になり、pack6 の log-increment が有限になる。
    # L_star_guess はあくまで表面 T_eff 計算にのみ使う。
    L_raw_total = float(L2[-1]) if L2[-1] > 0 else float(L_star_guess)
    if L_raw_total < 1e-10:
        L_raw_total = float(L_star_guess)
    # ell = L_raw / L_raw_total → ell[-1] = 1 が自然に成立
    L_est_init = L_raw_total
    log_L_est0 = np.log(L_est_init / L_sun)

    # T の外層改良 (表面 T_eff は L_star_guess から推定)
    L_guess_for_T = max(float(L_star_guess), L_est_init)
    T_eff_init = max((L_guess_for_T/(4*np.pi*R_star_guess**2*sigma_sb))**0.25, 1000.0)
    kap_es     = getattr(args, 'kappa_es_factor', 1.0) * 0.2*(1+X)
    P_surf_est = (2.0/3.0)*(G*M_star/R_star_guess**2)/max(kap_es,1e-10)
    lm_ratio   = max((L_guess_for_T/L_sun)/max(M_hat,1e-9), 1e-9)
    nab_out    = float(np.clip(0.28/max(lm_ratio**0.12,0.01), 0.17, 0.40))
    T_out      = np.maximum(T_eff_init*(P_i/max(P_surf_est,1.0))**nab_out, T_eff_init)
    logPle     = np.log10(np.maximum(P_i, 1e-80))
    logPc      = float(np.log10(max(P_i[0], 1.0)))
    f_out      = 1.0/(1.0+np.exp((logPle-(logPc-1.5))/0.5))
    T_i        = np.maximum(np.exp(f_out*np.log(np.maximum(T_out,T_eff_init))+
                                   (1-f_out)*np.log(np.maximum(T_i,T_eff_init))), T_eff_init)

    # 変換
    S_i   = np.log(np.maximum(r_i, 1e-99) / R_sun)
    # ell = L_raw / L_raw_total → 単調増加 [0,1]
    ell_i = np.maximum(L_i / max(L_est_init, 1e-99), 1e-14)
    p_i   = np.log(np.maximum(P_i, 1e-99) / P_REF)
    t_i   = np.log(np.maximum(T_i, 1e-99) / T_REF)

    # 単調性
    S_i   = np.maximum.accumulate(S_i)
    ell_i = np.maximum.accumulate(ell_i)
    ell_i[-1] = 1.0  # 境界条件
    for arr in (p_i, t_i):
        for k in range(len(arr)-2, -1, -1):
            if arr[k] < arr[k+1]+1e-10: arr[k] = arr[k+1]+1e-10

    # ── PP_CALIB 適用後の ell 再初期化 ───────────────────────────
    # Lane-Emden の L_raw は PP_CALIB=1 で計算されたが、
    # 現在の PP_CALIB で eps を再計算して ell を正確に初期化する。
    # これにより Rell 残差 (光度方程式) がほぼゼロから出発できる。
    P_i_phys  = P_REF * np.exp(np.clip(p_i,  -100, 20))
    T_i_phys  = T_REF * np.exp(np.clip(t_i,  -80,  10))
    rho_i_phys = density_from_PT(P_i_phys, T_i_phys)
    eps_i_curr = energy_generation(rho_i_phys, T_i_phys)
    q_m = q_mesh  # N 点のメッシュ
    dq_m = np.diff(q_m)
    Lc_curr = np.zeros(len(q_m))
    for k in range(1, len(q_m)):
        Lc_curr[k] = Lc_curr[k-1] + 0.5*(eps_i_curr[k-1]+eps_i_curr[k])*M_hat*M_sun*dq_m[k-1]
    Lt_curr = float(Lc_curr[-1])
    if Lt_curr > 0:
        ell_i = np.maximum(Lc_curr / Lt_curr, 1e-14)
        ell_i[-1] = 1.0
        log_L_est0 = np.log(Lt_curr / L_sun)

    # 表面値を BC と整合させる
    t_surf, p_surf, _, _, _, _ = surface_bc_values(S_i[-1], ell_i[-1], log_L_est0, M_hat_val, args)
    t_i[-1] = t_surf
    p_i[-1] = p_surf

    # 数値的な逸脱を防ぐため、各プロファイルを物理的な範囲にクリップ
    S_i   = np.clip(S_i,   -10.0, np.log(5.0))   # R: exp(-10)〜5 R_sun
    ell_i = np.clip(ell_i, 1e-14, 1.0)
    p_i   = np.clip(p_i,   -50.0, 30.0)
    t_i   = np.clip(t_i,   -30.0,  5.0)
    log_L_est0 = float(np.clip(log_L_est0, -12.0, 12.0))

    return S_i, ell_i, p_i, t_i, log_L_est0


def load_ph4_solution_as_ph6_init(npz_path, q_target, args, L_est_override=None):
    """
    Phase 4 の収束解 (.npz) を Phase 6 初期値に変換する。

    Phase 4: s = r/R_star, l = L/L_star (R_star = L_star = 太陽単位)
    Phase 6: S = log(r/R_sun), ell = L/L_est
    変換: S = log(s),  ell = l (L_star=L_est の場合)
    """
    d4 = np.load(npz_path, allow_pickle=True)
    q4 = d4['q']; s4 = d4['s']; l4 = d4['l']; p4 = d4['p']; t4 = d4['t']

    P_unit_ph4 = 2.5e17   # Phase 4 の P_unit (太陽)
    T_unit_ph4 = 1.57e7   # Phase 4 の T_unit (太陽)

    # 変換
    S4  = np.log(np.maximum(s4, 1e-30))  # s = r/R_sun → S = log(r/R_sun)
    ell4 = l4.copy()                     # l = L/L_star → ell = L/L_est (L_star=L_est)
    p4c  = p4 + np.log(P_unit_ph4 / P_REF)
    t4c  = t4 + np.log(T_unit_ph4 / T_REF)

    # L_est: Phase 4 で L_star = L_sun (太陽の場合) → log_L_est = 0
    log_L_est0 = 0.0 if L_est_override is None else np.log(L_est_override/L_sun)

    # q_target メッシュに補間 (メッシュ点数が異なる場合も含む)
    if len(q4) != len(q_target) or not np.allclose(q4, q_target):
        S4   = np.interp(q_target, q4, S4)
        ell4 = np.interp(q_target, q4, ell4)
        p4c  = np.interp(q_target, q4, p4c)
        t4c  = np.interp(q_target, q4, t4c)

    # 単調性を強制
    S4   = np.maximum.accumulate(S4)
    ell4 = np.maximum.accumulate(ell4); ell4[-1] = 1.0
    for arr in (p4c, t4c):
        for k in range(len(arr)-2, -1, -1):
            if arr[k] < arr[k+1]+1e-10: arr[k] = arr[k+1]+1e-10

    # 表面値を BC と整合させる
    t_surf, p_surf, _, _, _, _ = surface_bc_values(S4[-1], ell4[-1], log_L_est0, M_hat, args)
    t4c[-1] = t_surf; p4c[-1] = p_surf

    return S4, ell4, p4c, t4c, log_L_est0


# ══════════════════════════════════════════════════════
# 7.  Bounds / Solve
# ══════════════════════════════════════════════════════

def make_bounds6(N):
    """
    pack6 変数の下界・上界を返す。

    pack6 の変数構造:
      xs[0]     = S[0]                (中心 log-radius)
      xs[1:]    = log(diff(S))        (S の log-increment、通常は小さな負値)
      xell[0]   = ell[0]              (中心光度)
      xell[1:]  = log(diff(ell))      (ell の log-increment)
      xp[:-1]   = log(p[i]-p[i+1])   (圧力差のログ)
      xp[-1]    = p[-1]               (表面圧力)
      xt[:-1]   = log(t[i]-t[i+1])   (温度差のログ)
      xt[-1]    = t[-1]               (表面温度)
      log_L_est = スカラー

    注意: R_star (S[-1]) の制限は pack6 の変数 xs[N-1] = log(S[-1]-S[-2]) では
    直接制御できない (累積和で決まるため)。R_star の物理的制限は
    w_R_prior の soft constraint で行い、ここでは設定しない。
    """
    lo = np.full(4*N+1, -80.0); hi = np.full(4*N+1, 80.0)
    lo[0]     = -20.0;  hi[0]     = 5.0    # S[0] (中心位置)
    lo[N]     = -30.0;  hi[N]     = 0.01   # ell[0] (中心光度、<<1)
    lo[3*N-1] = -50.0;  hi[3*N-1] = 5.0    # p_surf (表面圧力)
    lo[4*N-1] = -30.0;  hi[4*N-1] = 5.0    # t_surf (表面温度)
    lo[4*N]   = -12.0;  hi[4*N]   = 12.0   # log_L_est
    return lo, hi


from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp


# ─── プロセス並列 Jacobian のサポート関数 ───────────────────────────────────
# 注意: Python の GIL のため、スレッド並列は残差関数に効果がない。
# 各プロセスは独立した GIL を持つのでプロセス並列が有効。
#
# 重要: ワーカーは "stellar_structure_phase6" という固定名でなく、
# __file__ の絶対パスからモジュールをロードする。
# これにより stellar_structure_phase6-5.py のようにリネームされた
# ファイルを実行しても正しい残差関数が使われる。

# モジュール自身のファイルパス（リネーム対応のため __file__ を使用）
_PH6_MODULE_FILE = os.path.abspath(__file__)


def _jac_worker_init(module_file, opal_path, X_val, Y_val, Z_val,
                     pp_calib, cno_calib, saha_args_dict):
    """
    各ワーカープロセスの初期化。
    importlib で実ファイルパスからモジュールをロードし
    OPAL・Saha テーブルを準備する。
    """
    import importlib.util, sys, builtins
    # ハイフン付きファイル名でも正しくロードできるよう spec を使う
    _spec = importlib.util.spec_from_file_location('_ph6_w', module_file)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    # グローバル状態を設定
    _mod.apply_composition(X_val, Y_val, Z_val)
    _mod.PP_CALIB  = pp_calib
    _mod.CNO_CALIB = cno_calib
    # OPAL warmup
    if opal_path and os.path.exists(opal_path):
        try:
            _mod._get_opal_table(opal_path, X_val, Z_val)
        except Exception:
            pass
    # Saha テーブル warmup (Phase 4 があれば)
    try:
        import stellar_structure_phase4 as _ph4
        import argparse as _ap
        _sa = _ap.Namespace(**saha_args_dict)
        _ph4.builtin_saha_nabla_ad_table2d(_sa)
    except Exception:
        pass
    # ワーカープロセスのグローバルにモジュール参照を保存
    builtins._ph6_worker_module = _mod


def _jac_col_worker(packed):
    """1列分の有限差分 Jacobian を計算するプロセスワーカー。"""
    j, h, X, q, args = packed
    import builtins
    _m6 = builtins._ph6_worker_module
    Xp = X.copy(); Xm = X.copy()
    Xp[j] += h;  Xm[j] -= h
    fp = _m6.residual_vector6(Xp, q, args)
    fm = _m6.residual_vector6(Xm, q, args)
    return j, (fp - fm) / (2.0 * h)


def _parallel_jacobian(residual_fn, X, args_tuple, n_workers, h=1e-7):
    """
    有限差分 Jacobian を ProcessPoolExecutor で並列計算する。

    Python GIL のため ThreadPool は残差関数に効果がない。
    プロセス並列では GIL が独立しており本質的な並列化が実現できる。

    起動コスト: 各プロセスで OPAL + Saha テーブルをロードするため
    初回は ~0.2s/プロセスのオーバーヘッドがある。
    Jacobian 1回の利益 (N=80, 16コア): 直列770ms → 約260ms (3倍速)。

    Parameters
    ----------
    n_workers : int
        プロセス数。0 以下または 1 なら直列計算。
    """
    q, args = args_tuple
    n = len(X)
    f0 = residual_fn(X, q, args)
    m  = len(f0)

    if n_workers <= 1:
        # 直列フォールバック
        J = np.empty((m, n))
        for j in range(n):
            Xp = X.copy(); Xm = X.copy()
            Xp[j] += h;   Xm[j] -= h
            J[:, j] = (residual_fn(Xp, q, args) - residual_fn(Xm, q, args)) / (2*h)
        return J

    # プロセス初期化用の引数を収集
    _opal_path = getattr(args, 'opal_table', None)
    _X  = getattr(args, 'X_init', 0.70)
    _Y  = getattr(args, 'Y_init', 0.28)
    _Z  = getattr(args, 'Z_init', 0.02)
    _saha_dict = {
        'ad_table_source': 'builtin-saha2d',
        'saha_n_logT': getattr(args, 'saha_n_logT', 25),
        'saha_n_logrho': getattr(args, 'saha_n_logrho', 18),
        'X_init': _X, 'Y_init': _Y, 'Z_init': _Z,
        'M_star_solar': 1.0, 'R_star_solar': 1.0, 'L_star_solar': 1.0,
        'opacity_source': getattr(args, 'opacity_source', 'opal'),
        'opal_X': _X, 'opal_Z': _Z,
        'opal_table': _opal_path or '',
        'kappa_es_factor': getattr(args, 'kappa_es_factor', 1.0),
        'kappa_kramers_factor': getattr(args, 'kappa_kramers_factor', 1.0),
        'kappa_hminus_factor': getattr(args, 'kappa_hminus_factor', 1.0),
        'opacity_lowT_guard': 'none',
    }

    tasks = [(j, h, X, q, args) for j in range(n)]
    J = np.empty((m, n))

    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_jac_worker_init,
            initargs=(_PH6_MODULE_FILE, _opal_path, _X, _Y, _Z,
                      PP_CALIB, CNO_CALIB, _saha_dict)
    ) as ex:
        for j_idx, col in ex.map(_jac_col_worker, tasks):
            J[:, j_idx] = col

    return J


def solve_star6(M_hat_val, args,
                R_star_guess=None, L_star_guess=None,
                X0=None, q_given=None):
    """
    Phase 6 メイン求解。M_hat_val [M_sun] の ZAMS を解く。
    R_star と L_star は固有値として出力される。
    """
    global M_hat, PP_CALIB, CNO_CALIB
    apply_composition(
        getattr(args,'X_init',0.70),
        getattr(args,'Y_init',0.28),
        getattr(args,'Z_init',0.02))
    M_hat = float(M_hat_val)

    # ── Saha テーブルの自動ロード (Phase 4 が使える場合) ──────────
    # これにより nabla_ad と opacity が正確になる。
    # 呼ばれるたびにロード済みならスキップ (キャッシュ済み)。
    if _HAVE_PH4:
        try:
            import argparse as _ap
            _saha_args = _ap.Namespace(
                ad_table_source='builtin-saha2d',
                saha_n_logT=getattr(args,'saha_n_logT',25),
                saha_n_logrho=getattr(args,'saha_n_logrho',18),
                X_init=getattr(args,'X_init',0.70),
                Y_init=getattr(args,'Y_init',0.28),
                Z_init=getattr(args,'Z_init',0.02),
                M_star_solar=1.0, R_star_solar=1.0, L_star_solar=1.0,
                opacity_source=getattr(args,'opacity_source','opal'),
                opal_X=getattr(args,'opal_X',0.70),
                opal_Z=getattr(args,'opal_Z',0.02),
                opal_table=getattr(args,'opal_table','GN93hz.dat'),
                kappa_es_factor=getattr(args,'kappa_es_factor',1.0),
                kappa_kramers_factor=getattr(args,'kappa_kramers_factor',1.0),
                kappa_hminus_factor=getattr(args,'kappa_hminus_factor',1.0),
                opacity_lowT_guard='none',
            )
            _ph4.builtin_saha_nabla_ad_table2d(_saha_args)
        except Exception:
            pass

    # ZAMS の初期推定値 (デフォルト: args から取得)
    L_zams = getattr(args, 'L_zams_solar', 0.72)
    R_zams = getattr(args, 'R_zams_solar', 0.87)
    if R_star_guess is None:
        R_star_guess = R_sun * (M_hat ** 0.8) * R_zams
    if L_star_guess is None:
        if M_hat <= 0.5:   L_star_guess = L_sun * L_zams * (M_hat / 1.0) ** 5.0
        elif M_hat <= 2.0: L_star_guess = L_sun * L_zams * (M_hat / 1.0) ** 4.0
        else:              L_star_guess = L_sun * L_zams * (M_hat / 1.0) ** 3.5

    N = getattr(args, 'n_mesh', 80)
    q = make_q_mesh(N, getattr(args,'q0',1e-8), getattr(args,'q_power',1.4))
    N = len(q)

    print("="*72)
    print(f"  Phase 6 ZAMS  M={M_hat:.3f} M_sun  "
          f"X={getattr(args,'X_init',0.70):.3f}  Z={getattr(args,'Z_init',0.02):.4f}")
    print(f"  PP_CALIB={PP_CALIB:.4f}  CNO_CALIB={CNO_CALIB:.4f}  T_REF={T_REF/1e6:.3f} MK")
    print("="*72)

    if X0 is not None and q_given is not None:
        S_p, ell_p, p_p, t_p, log_Le_p = unpack6(X0, len(q_given))
        S0   = np.interp(q, q_given, S_p)
        ell0 = np.interp(q, q_given, ell_p)
        p0   = np.interp(q, q_given, p_p)
        t0   = np.interp(q, q_given, t_p)
        S0   = np.maximum.accumulate(S0)
        ell0 = np.maximum.accumulate(ell0); ell0[-1] = 1.0
        for arr in (p0, t0):
            for k in range(N-2,-1,-1):
                if arr[k]<arr[k+1]+1e-10: arr[k]=arr[k+1]+1e-10
        t_surf, p_surf, _, _, _, _ = surface_bc_values(S0[-1], ell0[-1], log_Le_p, M_hat, args)
        t0[-1]=t_surf; p0[-1]=p_surf
        X0_run = pack6(S0, ell0, p0, t0, log_Le_p)
        print(f"  前解を補間して初期値設定 (homotopy)")
    else:
        # ── Phase 4 / Phase 6 の既存解を自動検索 ───────────────────
        # 優先順位: 1) 前回の ph6 収束解 (ph6_solar_sol.npz)
        #           2) sol_N80.npz (Phase 4 の標準解)
        # これにより初期残差が大幅に改善され、収束が安定する。
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _auto_candidates = [
            os.path.join(_script_dir, 'ph6_solar_sol.npz'),
            'ph6_solar_sol.npz',
            os.path.join(_script_dir, 'sol_N80.npz'),
            'sol_N80.npz',
            os.path.expanduser('~/sol_N80.npz'),
        ]
        _found_init = None
        for _cand in _auto_candidates:
            if os.path.exists(_cand):
                _found_init = _cand
                break

        # ph6_solar_sol.npz が偽解 (R>2 or T_c<8MK) なら無視して Phase 4 に切り替える
        if _found_init is not None and 'ph6' in os.path.basename(_found_init):
            try:
                _chk = np.load(_found_init, allow_pickle=True)
                _R_chk = float(_chk.get('R_solar', 99.0))
                _Tc_chk = float(_chk.get('T_c', 0.0))
                if _R_chk > 2.0 or _Tc_chk < 8e6:
                    print(f"  [警告] ph6_solar_sol.npz は物理的でない偽解 "
                          f"(R={_R_chk:.2f} R_sun, T_c={_Tc_chk/1e6:.1f} MK)")
                    print(f"  → 偽解を無視して Phase 4 解 (sol_N80.npz) から再出発します")
                    # Phase 4 候補のみ残す
                    _found_init = None
                    for _cand in _auto_candidates:
                        if os.path.exists(_cand) and 'ph6' not in os.path.basename(_cand):
                            _found_init = _cand
                            break
            except Exception:
                pass

        if _found_init is not None and M_hat_val == 1.0:
            _is_ph6_sol = 'ph6' in os.path.basename(_found_init)
            print(f"  既存解を自動使用 ({'Phase 6' if _is_ph6_sol else 'Phase 4'}): {_found_init}")
            if _is_ph6_sol:
                _dat = np.load(_found_init, allow_pickle=True)
                _X_s, _q_s = _dat['X'], _dat['q']
                _N_s = len(_q_s)
                _S_p,_,_p_p,_t_p,_ = unpack6(_X_s, _N_s)
                S0 = np.interp(q, _q_s, _S_p)
                p0 = np.interp(q, _q_s, _p_p)
                t0 = np.interp(q, _q_s, _t_p)
            else:
                S0, _, p0, t0, _ = load_ph4_solution_as_ph6_init(_found_init, q, args)
            # 現在の PP_CALIB で ell を再初期化
            _P0 = P_REF * np.exp(np.clip(p0, -100, 20))
            _T0 = T_REF * np.exp(np.clip(t0, -80, 10))
            _rho0 = density_from_PT(_P0, _T0)
            _eps0 = energy_generation(_rho0, _T0)
            _dq = np.diff(q)
            _Lc = np.zeros(N)
            for _i in range(1, N):
                _Lc[_i] = _Lc[_i-1] + 0.5*(_eps0[_i-1]+_eps0[_i])*M_hat*M_sun*_dq[_i-1]
            _Lt = float(_Lc[-1])

            # ── log_L_est の初期化 ────────────────────────────────
            # Phase 4 T,P プロファイルから積分した L_total (_Lt) はしばしば
            # ZAMS 目標値 (0.72 L☉) から大きくずれる (典型的に 1.0〜1.1 L☉)。
            # このずれが ML prior 残差を小さく見せ、ソルバーが別の偽解に
            # 落ちる原因になる。
            #
            # 解決策: log_L_est を ZAMS 目標値に強制設定する。
            # これにより:
            # 1. ML prior 残差 = 0 → ソルバーが正しい方向を向く
            # 2. ell × L_est = 0.72 L☉ × ell → 表面 BC 自然に成立
            # 3. Rell 残差は大きくなるが、T_c を少し下げれば解消できる
            #    (Phase 4 T_c=15.7 MK → ZAMS T_c≈15.2 MK へ)
            #
            # また、PP_CALIB を Phase 4 T,P と整合するよう一時的にスケールして
            # ell プロファイルの形状を正確にする。
            _L_zams_solar = getattr(args, 'L_zams_solar', 0.72)

            if _Lt > 0:
                ell0 = np.maximum(_Lc / _Lt, 1e-14); ell0[-1] = 1.0
                log_Le0 = np.log(_L_zams_solar)  # ML prior を 0 から開始
            else:
                ell0 = np.ones(N) * 1e-10; ell0[-1] = 1.0
                log_Le0 = np.log(_L_zams_solar)
            t_surf, p_surf, _, _, _, _ = surface_bc_values(S0[-1], 1.0, log_Le0, M_hat, args)
            p0[-1] = p_surf; t0[-1] = t_surf
            for k in range(N-2, -1, -1):
                if t0[k] < t0[k+1]+1e-10: t0[k] = t0[k+1]+1e-10
                if p0[k] < p0[k+1]+1e-10: p0[k] = p0[k+1]+1e-10
            S0    = np.clip(S0,    -10.0, np.log(5.0))
            ell0  = np.clip(ell0,  1e-14, 1.0)
            p0    = np.clip(p0,    -50.0, 30.0)
            t0    = np.clip(t0,    -30.0,  5.0)
            log_Le0 = float(np.clip(log_Le0, -12.0, 12.0))

            # ── 2段階ソルブ: Phase4 → ZAMS への橋渡し ───────────────
            # 偽解 (T_c≈10MK, L≈0.04) 回避策:
            #   Stage1 (w_ml=3.0, 200eval): 強い ML prior で log_L_est を
            #     ZAMS 値(-0.33)近傍に強制 → 低温偽解から脱出
            #   Stage2 (w_ml=0.3, 通常):    通常設定で ZAMS 解に収束
            # PP_CALIB は常に元の値 (0.3945) を使う
            _args_s1 = argparse.Namespace(**vars(args))
            _args_s1.w_ml_prior = 3.0   # 強い ML prior → L を 0.72 に誘導
            _args_s1.w_R_prior  = 5.0   # 強い R prior → Hayashi 発散防止 (R>2 R_sun を阻止)
            _lo_s1, _hi_s1 = make_bounds6(N)
            _X0_s1 = np.clip(pack6(S0, ell0, p0, t0, log_Le0), _lo_s1, _hi_s1)
            print("  Stage 1: w_ml=3.0, w_R=5.0 (400 eval) で ZAMS 分枝へ...")
            _sol_s1 = least_squares(
                residual_vector6, _X0_s1, args=(q, _args_s1),
                bounds=(_lo_s1, _hi_s1), method='trf', jac='2-point',
                tr_solver=getattr(args,'tr_solver','lsmr'), x_scale='jac',
                ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=400, verbose=0)
            _d_s1 = compute_derived6(_sol_s1, q, _args_s1)
            _ri_s1 = np.max(np.abs(_sol_s1.fun))
            print(f"  Stage 1: |res|={_ri_s1:.3e} R={_d_s1['R_solar']:.3f} "
                  f"L={_d_s1['L_solar']:.3f} T_c={_d_s1['T_c']/1e6:.2f}MK")
            if _d_s1['L_solar'] > 0.1 and _d_s1['T_c'] > 8e6 and _d_s1['R_solar'] < 2.0:
                X0_run = np.clip(_sol_s1.x, _lo_s1, _hi_s1)
            else:
                print("  [警告] Stage 1 が物理外 → Lane-Emden 初期値を使用")
                # Phase 4 ではなく Lane-Emden 初期モデルにフォールバック
                S0_le, ell0_le, p0_le, t0_le, lLe_le = make_initial_model6_envelope(
                    q, M_hat_val, args, R_star_guess, L_star_guess)
                X0_run = np.clip(pack6(S0_le, ell0_le, p0_le, t0_le, lLe_le), _lo_s1, _hi_s1)
        else:
            S0, ell0, p0, t0, log_Le0 = make_initial_model6_envelope(
                q, M_hat_val, args, R_star_guess, L_star_guess)
            X0_run = pack6(S0, ell0, p0, t0, log_Le0)

        X0_run = np.clip(X0_run, *make_bounds6(N))
        print(f"  初期: |res|∞={np.max(np.abs(residual_vector6(X0_run,q,args))):.3e}")

    r0 = residual_vector6(X0_run, q, args)
    print(f"  初期 |res|∞={np.max(np.abs(r0)):.3e}  |res|2={np.linalg.norm(r0):.3e}")

    # bounds を超えた初期値をクリップ（ローカル環境での数値誤差対策）
    _lo, _hi = make_bounds6(N)
    X0_run = np.clip(X0_run, _lo, _hi)

    # ── N > 80 の場合: N=80 粗解が存在しなければ先行生成 ─────────────
    # N=120 での Stage 1 失敗を防ぐ。N=80 の解を初期値として使うと
    # N=120 でも Stage 1 不要で収束しやすい。
    _have_coarse = False
    if N > 80 and M_hat_val == 1.0:
        _coarse_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'ph6_solar_sol.npz')
        if not os.path.exists(_coarse_path):
            print(f"  N={N} の初期解なし → N=80 で粗解を生成中...")
            import argparse as _ap2
            _args_c = _ap2.Namespace(**vars(args)); _args_c.n_mesh = 80
            _q80 = make_q_mesh(80)
            # N=80 での Stage 1 (w_ml=3, w_R=5)
            _args_c1 = _ap2.Namespace(**vars(_args_c))
            _args_c1.w_ml_prior = 3.0; _args_c1.w_R_prior = 5.0
            _S80,_,_p80,_t80,_ = load_ph4_solution_as_ph6_init(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sol_N80.npz'),
                _q80, _args_c)
            _P80=P_REF*np.exp(np.clip(_p80,-100,20))
            _T80=T_REF*np.exp(np.clip(_t80,-80,10))
            _r80=density_from_PT(_P80,_T80); _e80=energy_generation(_r80,_T80)
            _N80=len(_q80); _dq80=np.diff(_q80); _Lc80=np.zeros(_N80)
            for _i in range(1,_N80): _Lc80[_i]=_Lc80[_i-1]+0.5*(_e80[_i-1]+_e80[_i])*M_hat*M_sun*_dq80[_i-1]
            _Lt80=float(_Lc80[-1]); _en80=np.maximum(_Lc80/max(_Lt80,1e-99),1e-14); _en80[-1]=1.0
            _lLe80=np.log(getattr(args,'L_zams_solar',0.72))
            _ts80,_ps80,_,_,_,_=surface_bc_values(_S80[-1],1.0,_lLe80,1.0,_args_c)
            _p80[-1]=_ps80; _t80[-1]=_ts80
            _S80=np.clip(_S80,-10,np.log(5)); _en80=np.clip(_en80,1e-14,1.0)
            _p80=np.clip(_p80,-50,30); _t80=np.clip(_t80,-30,5)
            for _k in range(_N80-2,-1,-1):
                if _t80[_k]<_t80[_k+1]+1e-10: _t80[_k]=_t80[_k+1]+1e-10
                if _p80[_k]<_p80[_k+1]+1e-10: _p80[_k]=_p80[_k+1]+1e-10
            _lo80,_hi80=make_bounds6(_N80)
            _X80=np.clip(pack6(_S80,_en80,_p80,_t80,_lLe80),_lo80,_hi80)
            _sol80=least_squares(residual_vector6, _X80, args=(_q80,_args_c1),
                                 bounds=(_lo80,_hi80), method='trf', jac='2-point',
                                 ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=400, verbose=0)
            _d80=compute_derived6(_sol80,_q80,_args_c1)
            print(f"  N=80 粗解: R={_d80['R_solar']:.3f} L={_d80['L_solar']:.3f} T_c={_d80['T_c']/1e6:.1f}MK")
            if _d80['R_solar'] < 2.0 and _d80['T_c'] > 8e6:
                # 粗解を補間して N の初期値に
                _S80r,_,_p80r,_t80r,_=unpack6(_sol80.x, _N80)
                S0_c=np.interp(q,_q80,_S80r); p0_c=np.interp(q,_q80,_p80r); t0_c=np.interp(q,_q80,_t80r)
                _P_c=P_REF*np.exp(np.clip(p0_c,-100,20)); _T_c=T_REF*np.exp(np.clip(t0_c,-80,10))
                _r_c=density_from_PT(_P_c,_T_c); _e_c=energy_generation(_r_c,_T_c)
                _dqc=np.diff(q); _Lcc=np.zeros(N)
                for _i in range(1,N): _Lcc[_i]=_Lcc[_i-1]+0.5*(_e_c[_i-1]+_e_c[_i])*M_hat*M_sun*_dqc[_i-1]
                _Ltc=float(_Lcc[-1]); _enc=np.maximum(_Lcc/max(_Ltc,1e-99),1e-14); _enc[-1]=1.0
                _lLec=np.log(getattr(args,'L_zams_solar',0.72))
                _ts_c,_ps_c,_,_,_,_=surface_bc_values(S0_c[-1],1.0,_lLec,1.0,args)
                p0_c[-1]=_ps_c; t0_c[-1]=_ts_c
                S0_c=np.clip(S0_c,-10,np.log(5)); _enc=np.clip(_enc,1e-14,1.0)
                p0_c=np.clip(p0_c,-50,30); t0_c=np.clip(t0_c,-30,5)
                for _k in range(N-2,-1,-1):
                    if t0_c[_k]<t0_c[_k+1]+1e-10: t0_c[_k]=t0_c[_k+1]+1e-10
                    if p0_c[_k]<p0_c[_k+1]+1e-10: p0_c[_k]=p0_c[_k+1]+1e-10
                X0_run = np.clip(pack6(S0_c,_enc,p0_c,t0_c,_lLec), _lo, _hi)
                _have_coarse = True
                print(f"  N=80 → N={N} に補間した初期値を使用")

    # ── メインソルブ: 適応的 w_ml_prior ─────────────────────────────
    # Stage 1 または粗解が L=0.72 から大きくずれている場合は、
    # w_ml_prior を 1.5 に強化してメインソルブで L が下がらないようにする。
    # L ≈ 0.72 のとき (ML prior≈0) は通常設定と同じ振る舞いをする。
    _args_main = argparse.Namespace(**vars(args))
    _r0_main = np.max(np.abs(residual_vector6(X0_run, q, args)))
    _S_tmp,_e_tmp,_,_,_lLe_tmp = unpack6(X0_run, N)
    _L_current = float(np.exp(float(_lLe_tmp)) * _e_tmp[-1])  # 現在の L_star 推定
    if _L_current < 0.5:
        _args_main.w_ml_prior = 2.0   # L が大きく外れ → 強い引力
    elif _L_current < 0.65:
        _args_main.w_ml_prior = 1.5   # L がやや外れ → 中程度の引力
    else:
        _args_main.w_ml_prior = max(getattr(args,'w_ml_prior',0.3), 0.5)
    print(f"  メインソルブ: w_ml={_args_main.w_ml_prior:.1f}  初期|res|∞={_r0_main:.3e}")

    n_workers = int(getattr(args, 'n_jac_workers', 0))
    if n_workers == 0:
        # 自動: プロセス起動コスト (~0.2s/プロセス) を考慮して
        # CPU 数の半分をデフォルトにする。1コアなら直列。
        cpu = _mp.cpu_count() or 1
        n_workers = max(1, cpu // 2) if cpu > 2 else 1
    print(f"  並列 Jacobian: n_workers={n_workers}  (cpu_count={_mp.cpu_count()})")

    if n_workers > 1:
        # 並列 Jacobian を使う（スレッド並列）
        # scipy は jac(X, *args) と呼び出すため *_ で余分な引数を吸収する
        def _jac_fn(X, *_):
            return _parallel_jacobian(residual_vector6, X, (q, _args_main), n_workers)

        sol = least_squares(
            residual_vector6, X0_run, args=(q, _args_main),
            jac=_jac_fn,
            bounds=make_bounds6(N), method='trf',
            tr_solver=getattr(args,'tr_solver','lsmr'), x_scale='jac',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=getattr(args, 'max_nfev', 2000), verbose=1)
    else:
        sol = least_squares(
            residual_vector6, X0_run, args=(q, _args_main),
            bounds=make_bounds6(N), method='trf', jac='2-point',
            tr_solver=getattr(args,'tr_solver','lsmr'), x_scale='jac',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=getattr(args, 'max_nfev', 2000), verbose=1)

    d = compute_derived6(sol, q, args)  # 最終評価は元の args で
    r_inf = np.max(np.abs(sol.fun))
    print("-"*72)
    print(f"  cost={sol.cost:.3e}  |res|∞={r_inf:.3e}")
    print(f"  R_star={d['R_solar']:.4f} R_sun  L_star={d['L_solar']:.4e} L_sun")

    # ── 収束解を自動保存 (次回の初期値として使う) ───────────────
    # 条件: M=1 M_sun かつ以下をすべて満たす
    #   ① |res|∞ < 0.7  (暫定解も保存して次回収束に活用)
    #   ② R_star < 2 R_sun  (Hayashi 偽解 R~3+ R_sun を除外)
    #   ③ T_c > 8 MK        (低温偽解 T_c~4 MK を除外)
    # 保存先: スクリプトと同じディレクトリの ph6_solar_sol.npz
    _is_physical = (d['R_solar'] < 2.0 and d['T_c'] > 8e6)
    if M_hat_val == 1.0 and r_inf < 0.7 and _is_physical:
        _save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'ph6_solar_sol.npz')
        try:
            np.savez(_save_path, X=sol.x, q=q, PP_CALIB=PP_CALIB,
                     R_solar=d['R_solar'], L_solar=d['L_solar'],
                     T_c=d['T_c'], rho_c=d['rho_c'],
                     res_inf=r_inf, n_mesh=len(q))
            _status = "収束" if r_inf < 0.05 else "暫定"
            print(f"  [{_status}] 解を保存: {_save_path}")
            print(f"  (N={len(q)}, |res|∞={r_inf:.3e}  次回の初期値として使用)")
        except Exception as e:
            print(f"  [警告] 解の保存に失敗: {e}")
            print(f"  保存先: {_save_path}")
    elif M_hat_val == 1.0 and not _is_physical:
        print(f"  [スキップ] 物理的でない解は保存しない "
              f"(R={d['R_solar']:.2f} R_sun, T_c={d['T_c']/1e6:.1f} MK)")
    print(f"  T_c={d['T_c']/1e6:.3f} MK  rho_c={d['rho_c']:.2f} g/cm3")
    print(f"  T_eff={d['T_eff']:.1f} K  CNO={d['cno_frac']*100:.2f}%")
    print(f"  L_mass/L_star={d['L_mass_solar']/max(d['L_solar'],1e-9):.4f}  (整合: 1.0)")
    print("="*72)
    return sol, q, d


# ══════════════════════════════════════════════════════
# 8.  診断
# ══════════════════════════════════════════════════════

def compute_derived6(sol, q, args):
    N = len(q)
    S, ell, p, t, log_L_est = unpack6(sol.x, N)
    r, m, P, L, T, rho = physical6(q, S, p, ell, t, log_L_est)

    R_star = R_sun * np.exp(S[-1])
    L_star = L_sun * np.exp(log_L_est) * ell[-1]

    eps_X, eps_Y, eps_Z, eps_mu, eps_XCNO = _comp_at(q)
    eps = energy_generation(rho, T, X_loc=eps_X, XCNO_loc=eps_XCNO)
    eps_pp, eps_cno = energy_generation_components(rho, T, X_loc=eps_X,
                                                   XCNO_loc=eps_XCNO)
    kap = opacity(rho, T, args, X_loc=eps_X, Z_loc=eps_Z)
    nabla, nr, na = choose_nabla(q, S, p, ell, t, log_L_est, args)

    T_eff_val = (L_star/(4*np.pi*R_star**2*sigma_sb))**0.25
    cno_frac  = float(np.trapezoid(eps_cno*rho, x=r) /
                      max(np.trapezoid(eps*rho, x=r), 1e-99))
    L_mass    = float(M_hat * M_sun * np.trapezoid(eps, x=q))
    t_surf, p_surf, _, _, _, _ = surface_bc_values(S[-1], ell[-1], log_L_est, M_hat, args)

    return dict(
        q=q, S=S, ell=ell, r=r, m=m, P=P, L=L, T=T, rho=rho,
        eps=eps, eps_pp=eps_pp, eps_cno=eps_cno, kap=kap,
        nabla=nabla, nabla_rad=nr, nabla_ad=na,
        R_star=R_star, L_star=L_star,
        R_solar=R_star/R_sun, L_solar=L_star/L_sun, M_solar=M_hat,
        T_eff=T_eff_val, T_c=float(T[0]), rho_c=float(rho[0]), P_c=float(P[0]),
        cno_frac=float(np.clip(cno_frac,0,1)),
        L_mass=L_mass, L_mass_solar=L_mass/L_sun,
        PP_CALIB=PP_CALIB, CNO_CALIB=CNO_CALIB,
        log_L_est=log_L_est, t_surf=t_surf, p_surf=p_surf,
    )


def print_summary6(sol, d, args, t_start=None, t_end=None):
    """
    解のサマリーを表示する。

    t_start, t_end: datetime オブジェクト（計算開始・終了時刻）。
    渡された場合は時刻・所要時間を出力する。
    """
    r_inf = np.max(np.abs(sol.fun))
    conv = "【収束✓】" if r_inf<0.01 else "【ほぼ収束】" if r_inf<0.1 else "【警告: 未収束】"
    print(); print("="*72)
    print(f"  Phase 6 ZAMS サマリー  {conv}  |res|∞={r_inf:.3e}")
    print("="*72)
    print(f"  入力: M={d['M_solar']:.3f} M_sun  "
          f"X={getattr(args,'X_init',0.70):.3f}  Z={getattr(args,'Z_init',0.02):.4f}")
    print(f"  出力 (固有値):")
    print(f"    R_star = {d['R_solar']:.4f} R_sun")
    print(f"    L_star = {d['L_solar']:.4e} L_sun")
    print(f"    T_eff  = {d['T_eff']:.1f} K")
    print(f"  中心:")
    print(f"    T_c    = {d['T_c']/1e6:.3f} MK")
    print(f"    rho_c  = {d['rho_c']:.2f} g/cm3")
    print(f"    P_c    = {d['P_c']:.3e} dyne/cm2")
    print(f"  エネルギー:")
    print(f"    L_star       = {d['L_solar']:.4e} L_sun")
    print(f"    L_mass (∫εdm)= {d['L_mass_solar']:.4e} L_sun "
          f"[比: {d['L_mass_solar']/max(d['L_solar'],1e-9):.4f}]")
    print(f"    CNO 分率     = {d['cno_frac']:.4f}")
    # ── 対流域の検出・表示 ──────────────────────────────
    unstable = d['nabla_rad'] > d['nabla_ad']
    if not np.any(unstable):
        print("  対流域: なし (全域 放射平衡)")
    else:
        # 連続する不安定領域を個別のゾーンに分割する
        zones = []
        in_zone = False
        for i, u in enumerate(unstable):
            if u and not in_zone:
                z_start = i
                in_zone = True
            elif not u and in_zone:
                zones.append((z_start, i - 1))
                in_zone = False
        if in_zone:
            zones.append((z_start, len(unstable) - 1))

        q_arr = d['q']; r_arr = d['r']
        print(f"  対流域: {len(zones)} ゾーン")
        for k, (i0, i1) in enumerate(zones):
            q0, q1 = q_arr[i0], q_arr[i1]
            r0, r1 = r_arr[i0] / R_sun, r_arr[i1] / R_sun
            nr0, nr1 = d['nabla_rad'][i0], d['nabla_rad'][i1]
            na0, na1 = d['nabla_ad'][i0],  d['nabla_ad'][i1]
            n_pts = i1 - i0 + 1
            label = "対流コア" if i0 == 0 else ("対流エンベロープ" if i1 == len(unstable)-1 else f"中間対流層")
            print(f"    ゾーン{k+1} [{label}]  {n_pts} メッシュ点")
            print(f"      q : {q0:.5f} → {q1:.5f}")
            print(f"      r : {r0:.4f} → {r1:.4f}  R_sun")
            print(f"      ∇_rad: {nr0:.4f} → {nr1:.4f}  |  ∇_ad: {na0:.4f} → {na1:.4f}")

    if t_start is not None and t_end is not None:
        elapsed = (t_end - t_start).total_seconds()
        print(f"  計算開始: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  計算終了: {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  所要時間: {elapsed:.3f} 秒")
    print("="*72)


# ─────────────────────────────────────────────────────
# テキストテーブル出力
# ─────────────────────────────────────────────────────

def _select_table_rows(N, target_min=20, target_max=25):
    """
    N 点のメッシュから先頭・末尾を含む 20–25 行を選ぶインデックスを返す。
    中間は対数的に等間隔に間引く。
    """
    if N <= target_max:
        return list(range(N))
    # 先頭・末尾は必ず含む
    # 中間を target_min-2 〜 target_max-2 個選ぶ（対数等間隔）
    n_mid = (target_min + target_max) // 2 - 2   # ≈ 21
    mid_idx = np.unique(np.round(
        np.logspace(0, np.log10(N - 2), n_mid + 2)
    ).astype(int))
    mid_idx = mid_idx[(mid_idx > 0) & (mid_idx < N - 1)]
    # target_max-2 個に絞る
    if len(mid_idx) > target_max - 2:
        step = len(mid_idx) / (target_max - 2)
        mid_idx = mid_idx[np.round(np.arange(target_max - 2) * step).astype(int)]
    idx = np.concatenate([[0], mid_idx, [N - 1]])
    return sorted(set(int(i) for i in idx))


def format_table6(d, mode='s'):
    """
    計算結果をテキストテーブルとして返す（文字列）。

    mode='s' : 間引き20–25行、フォーマット {:.2e}
    mode='l' : 全メッシュ、フォーマット {:.3e}

    カラム: r/R_sun, M/M_sun, T[K], P[dyn cm-2], rho[g cm-3], L/L_sun, rad/c
      rad/c = 'r'（放射平衡）または 'c'（対流不安定）
    """
    r    = d['r'] / R_sun
    m    = d['m'] / M_sun
    T    = d['T']
    P    = d['P']
    rho  = d['rho']
    L    = d['L'] / L_sun
    unstable = d['nabla_rad'] > d['nabla_ad']   # True → 対流
    N    = len(r)

    if mode == 's':
        rows = _select_table_rows(N)
        fmt  = lambda x: f"{x:.2e}"
    else:
        rows = list(range(N))
        fmt  = lambda x: f"{x:.3e}"

    header = (
        f"{'r/R_sun':>12}  {'M/M_sun':>12}  "
        f"{'T[K]':>12}  {'P[dyn cm-2]':>12}  "
        f"{'rho[g cm-3]':>12}  {'L/L_sun':>12}  {'rad/c':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for i in rows:
        rc = "c" if unstable[i] else "r"
        lines.append(
            f"{fmt(r[i]):>12}  {fmt(m[i]):>12}  "
            f"{fmt(T[i]):>12}  {fmt(P[i]):>12}  "
            f"{fmt(rho[i]):>12}  {fmt(L[i]):>12}  {rc:>5}"
        )
    return "\n".join(lines)



def write_table6(d, sol, args, t_start=None, t_end=None,
                 mode='s', outfile=None):
    """
    テキストテーブルをファイル（または stdout）に書き出す。

    mode : 's' = small (間引き), 'l' = large (全行)
    outfile : None なら stdout に出力
    """
    from datetime import datetime as _dt

    title_line = (
        f"# Phase 6 ZAMS テーブル出力\n"
        f"# M={d['M_solar']:.4f} M_sun  "
        f"X={getattr(args,'X_init',0.70):.3f}  "
        f"Z={getattr(args,'Z_init',0.02):.4f}\n"
        f"# R={d['R_solar']:.4f} R_sun  "
        f"L={d['L_solar']:.4e} L_sun  "
        f"T_eff={d['T_eff']:.1f} K\n"
        f"# T_c={d['T_c']/1e6:.3f} MK  "
        f"rho_c={d['rho_c']:.4f} g/cm3  "
        f"P_c={d['P_c']:.4e} dyne/cm2\n"
        f"# CNO分率={d['cno_frac']:.4f}  "
        f"|res|∞={np.max(np.abs(sol.fun)):.3e}"
    )
    if t_start is not None and t_end is not None:
        elapsed = (t_end - t_start).total_seconds()
        timing_line = (
            f"\n# 計算開始: {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# 計算終了: {t_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# 所要時間: {elapsed:.3f} 秒"
        )
    else:
        timing_line = ""

    row_label = "全メッシュ" if mode == 'l' else f"間引き（{len(_select_table_rows(len(d['r'])))}行）"
    mode_line = f"# テーブル行数モード: {mode} ({row_label})"

    table_str = format_table6(d, mode=mode)
    full_text = "\n".join([title_line, timing_line, mode_line, "", table_str]) + "\n"

    if outfile is None:
        print(full_text)
    else:
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"  テーブル出力: {outfile}")



# ══════════════════════════════════════════════════════
# 9.  太陽較正
# ══════════════════════════════════════════════════════

def calibrate_solar6(args, max_iter=6, tol=0.02, ph4_init=None):
    """
    PP_CALIB / CNO_CALIB を太陽 ZAMS モデルで較正する。

    較正目標: M=1 M☉ での ZAMS 光度・半径
      L_ZAMS = 0.72 L☉  (現在の太陽の 72%、ゼロ年齢主系列)
      R_ZAMS = 0.87 R☉  (現在の太陽の 87%)
    """
    global PP_CALIB, CNO_CALIB
    # 較正目標値を args から取得 (デフォルト: ZAMS 値)
    L_target = getattr(args, 'L_zams_solar', 0.72)
    R_target = getattr(args, 'R_zams_solar', 0.87)

    print(); print("="*72)
    print(f"  太陽 ZAMS 較正  目標: L={L_target:.3f} L☉  R={R_target:.3f} R☉")
    print("="*72)
    args.X_init=0.70; args.Y_init=0.28; args.Z_init=0.02
    sol_prev=None; q_prev=None

    for it in range(max_iter):
        print(f"\n  Iter {it+1}: PP={PP_CALIB:.4f}  CNO={CNO_CALIB:.4f}")
        if sol_prev is not None:
            sol,q,d = solve_star6(1.0, args, X0=sol_prev.x, q_given=q_prev)
        elif ph4_init and os.path.exists(ph4_init):
            q = make_q_mesh(getattr(args,'n_mesh',80))
            apply_composition(args.X_init,args.Y_init,args.Z_init)
            global M_hat; M_hat=1.0
            S0,ell0,p0,t0,lLe0 = load_ph4_solution_as_ph6_init(ph4_init, q, args)
            X0ph4 = pack6(S0,ell0,p0,t0,lLe0)
            sol,q,d = solve_star6(1.0, args, X0=X0ph4, q_given=q)
        else:
            sol,q,d = solve_star6(1.0, args,
                                  R_star_guess=R_target*R_sun,
                                  L_star_guess=L_target*L_sun)

        L_model = d['L_solar']
        R_model = d['R_solar']
        print(f"  → L={L_model:.4f} L☉  R={R_model:.4f} R☉  T_c={d['T_c']/1e6:.3f} MK")

        if abs(L_model - L_target) < tol:
            print(f"  較正完了! PP={PP_CALIB:.4f}  CNO={CNO_CALIB:.4f}")
            break

        # PP_CALIB を L_target/L_model の比で更新
        factor    = L_target / max(L_model, 1e-10)
        PP_CALIB  = float(np.clip(PP_CALIB * factor, 0.01, 20.0))
        CNO_CALIB = float(np.clip(CNO_CALIB * factor, 0.01, 20.0))
        sol_prev=sol; q_prev=q

    return sol, q, d


# ══════════════════════════════════════════════════════
# 10.  質量掃引
# ══════════════════════════════════════════════════════

def _solve_one_mass(args_packed):
    """
    質量掃引の1点を計算するワーカー関数 (ProcessPoolExecutor 用)。

    グローバル変数は使えないため、必要な状態を args_packed に詰める。
    """
    M_tgt, M_base, X_base, q_base, pp_calib, cno_calib, Xc, Yc, Zc, args_ns = args_packed
    # モジュールを再インポート（プロセス分離のため）
    import stellar_structure_phase6 as _ph6
    _ph6.PP_CALIB  = pp_calib
    _ph6.CNO_CALIB = cno_calib
    _ph6.apply_composition(Xc, Yc, Zc)
    try:
        sol, q, d = _ph6.solve_star6(M_tgt, args_ns, X0=X_base, q_given=q_base)
        return (M_tgt, True, sol.x, q, d)
    except Exception as e:
        return (M_tgt, False, None, None, str(e))


def mass_sweep6(M_targets, args, start_sol=None, start_q=None):
    """
    複数の質量点で ZAMS を解く (連続法 + 並列化)。

    並列化戦略:
    - M=1 の収束解を起点に連続法でホモトピー初期値を生成
    - 独立した質量点（起点が同じもの）は ProcessPoolExecutor で並列計算
    - n_sweep_workers=0 (デフォルト) で自動的に CPU 数を使用
    """
    print(); print("="*72)
    print(f"  質量連続法  対象: {[f'{m:.2f}' for m in M_targets]}")
    print("="*72)

    if start_sol is None:
        print("\n[基準: M=1.0 M_sun]")
        sol_sun, q_sun, d_sun = solve_star6(1.0, args)
    else:
        sol_sun, q_sun = start_sol, start_q
        d_sun = compute_derived6(sol_sun, q_sun, args)

    results = [(1.0, sol_sun, q_sun, d_sun)]

    # ── ホモトピー初期値の生成 (直列) ──────────────────────────
    # 各 M に対して「どの M を起点にするか」と「ホモトピー列」を決める
    tasks = []
    for M_tgt in sorted(M_targets, key=lambda m: abs(m - 1.0)):
        if abs(M_tgt - 1.0) < 1e-6:
            continue
        best = min(results, key=lambda x: abs(x[0] - M_tgt))
        M_b, sol_b, q_b, _ = best
        steps = np.linspace(M_b, M_tgt, max(2, int(np.ceil(abs(M_tgt - M_b)/0.3)) + 1))[1:]

        # 中間ステップは直列で連続法
        sc, qc = sol_b, q_b
        ok = True
        for M_s in steps[:-1]:
            try:
                sn, qn, dn = solve_star6(M_s, args, X0=sc.x, q_given=qc)
                sc, qc = sn, qn
            except Exception as e:
                print(f"  エラー (ホモトピー) M={M_s:.3f}: {e}")
                ok = False; break
        if ok:
            tasks.append((M_tgt, float(steps[-2] if len(steps) > 1 else M_b),
                          sc.x, qc))

    # ── 最終ステップを並列計算 ──────────────────────────────────
    n_sweep = int(getattr(args, 'n_sweep_workers', 0))
    if n_sweep == 0:
        n_sweep = max(1, min(_mp.cpu_count() or 1, len(tasks)))

    if n_sweep > 1 and len(tasks) > 1:
        print(f"\n  最終ステップを並列計算: {n_sweep} workers, {len(tasks)} 点")
        packed = [
            (M_tgt, M_base, X_base, q_base,
             PP_CALIB, CNO_CALIB, X, Y, Z, args)
            for M_tgt, M_base, X_base, q_base in tasks
        ]
        with ProcessPoolExecutor(max_workers=n_sweep) as ex:
            futures = list(ex.map(_solve_one_mass, packed))
        for item in futures:
            M_tgt, ok, X_sol, q_sol, d_or_err = item
            if ok:
                from types import SimpleNamespace
                _sol = SimpleNamespace(x=X_sol, fun=np.zeros(1))
                _d   = d_or_err
                results.append((M_tgt, _sol, q_sol, _d))
                print(f"  [並列] M={M_tgt:.3f}: R={_d['R_solar']:.3f} L={_d['L_solar']:.3f}")
            else:
                print(f"  [並列] M={M_tgt:.3f}: エラー {d_or_err}")
    else:
        # 直列フォールバック
        for M_tgt, M_base, X_base, q_base in tasks:
            try:
                sn, qn, dn = solve_star6(M_tgt, args, X0=X_base, q_given=q_base)
                results.append((M_tgt, sn, qn, dn))
            except Exception as e:
                print(f"  エラー M={M_tgt:.3f}: {e}")

    return results


# ══════════════════════════════════════════════════════
# 11.  プロット
# ══════════════════════════════════════════════════════


def plot_solution6(sol, q, d, args, outfile="ph6_sol.png"):
    """Plot ZAMS structure profiles (all labels in English)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    r = d['r'] / R_sun

    axes[0,0].semilogy(r, d['T']/1e6, 'r-')
    axes[0,0].set(xlabel='r / R_sun', ylabel='T [MK]', title='Temperature')
    axes[0,1].semilogy(r, d['rho'], 'b-')
    axes[0,1].set(xlabel='r / R_sun', ylabel='density [g/cm3]', title='Density')
    axes[0,2].plot(r, d['L']/L_sun, 'g-')
    axes[0,2].axhline(d['L_solar'], color='k', ls='--', lw=0.8)
    axes[0,2].set(xlabel='r / R_sun', ylabel='L / L_sun', title='Luminosity')

    axes[1,0].semilogy(r, d['eps_pp'],  'b-', label='pp')
    axes[1,0].semilogy(r, np.maximum(d['eps_cno'], 1e-30), 'r-', label='CNO')
    axes[1,0].legend(fontsize=8)
    axes[1,0].set(xlabel='r / R_sun', ylabel='eps [erg/g/s]', title='Energy generation')

    axes[1,1].plot(r, d['nabla_rad'], 'r-', label='nabla_rad')
    axes[1,1].plot(r, d['nabla_ad'],  'b-', label='nabla_ad')
    axes[1,1].plot(r, d['nabla'],     'k-', lw=0.8, label='nabla_eff')
    axes[1,1].legend(fontsize=8)
    axes[1,1].set(xlabel='r / R_sun', ylabel='nabla', ylim=(-0.05, 0.6),
                  title='Temperature gradient')

    axes[1,2].semilogy(np.abs(sol.fun) + 1e-15, 'k-', lw=0.5)
    axes[1,2].set(xlabel='Residual index', ylabel='|residual|',
                  title=f'Residuals  |res|inf={np.max(np.abs(sol.fun)):.2e}')

    fig.suptitle(
        f"Phase 6 ZAMS:  M={d['M_solar']:.3f} Msun  "
        f"R={d['R_solar']:.3f} Rsun  "
        f"L={d['L_solar']:.3e} Lsun  "
        f"T_c={d['T_c']/1e6:.2f} MK  CNO={d['cno_frac']:.3f}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {outfile}")


def plot_sweep6(results, outfile="ph6_sweep.png"):
    """Plot mass-sweep results (all labels in English)."""
    Ms   = np.array([r[0]             for r in results])
    idx  = np.argsort(Ms); Ms = Ms[idx]
    Rs   = np.array([r[3]['R_solar']  for r in results])[idx]
    Ls   = np.array([r[3]['L_solar']  for r in results])[idx]
    Tcs  = np.array([r[3]['T_c']/1e6  for r in results])[idx]
    cnos = np.array([r[3]['cno_frac'] for r in results])[idx]
    Teffs= np.array([r[3]['T_eff']    for r in results])[idx]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Phase 6 ZAMS  Mass Sweep")

    axes[0,0].plot(Ms, Tcs,  'ro-')
    axes[0,0].set(xlabel='M / M_sun', ylabel='T_c [MK]', title='Central temperature')
    axes[0,1].plot(Ms, Rs,   'bs-')
    axes[0,1].set(xlabel='M / M_sun', ylabel='R / R_sun', title='Radius')
    axes[0,2].semilogy(Ms, Ls, 'g^-')
    axes[0,2].set(xlabel='M / M_sun', ylabel='L / L_sun', title='Luminosity')
    axes[1,0].plot(Ms, cnos, 'kD-')
    axes[1,0].axhline(0.5, color='r', ls='--', lw=1)
    axes[1,0].set(xlabel='M / M_sun', ylabel='CNO / total', title='CNO fraction')
    axes[1,1].plot(Teffs[::-1], Ls, 'ko-')
    axes[1,1].set_yscale('log')
    axes[1,1].invert_xaxis()
    axes[1,1].set(xlabel='T_eff [K]', ylabel='L / L_sun', title='HR diagram')
    Mref = np.linspace(Ms.min(), Ms.max(), 50)
    axes[1,2].loglog(Ms, Ls, 'g-', label='L (computed)')
    axes[1,2].loglog(Mref, Mref**4, 'g--', lw=1, label='L~M^4')
    axes[1,2].loglog(Ms, Rs, 'b-', label='R (computed)')
    axes[1,2].loglog(Mref, Mref**0.8, 'b--', lw=1, label='R~M^0.8')
    axes[1,2].legend()
    axes[1,2].set(xlabel='M / M_sun', title='Scaling relations')
    plt.tight_layout()
    plt.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Sweep plot saved: {outfile}")

def build_default_args():
    return argparse.Namespace(
        X_init=0.70, Y_init=0.28, Z_init=0.02,
        n_mesh=80, q0=1e-8, q_power=1.4,
        weight_structure=1.0, weight_bc=5.0,   # BC を強めに (ell[-1]→1)
        rs_floor=1e-8, rs_relative_weight=1.0, outer_rs_raw_weight=1.0,
        outer_q_start=0.5, outer_q_width=0.1,
        nabla_hard_cap=10.0, nabla_center_cap=0.4,  # 外側 cap を緩める
        q_nabla_center=0.002, q_nabla_width=0.002,
        schwarz_smooth_eps=0.03,   # Schwarzschild 判定の tanh 平滑化幅
        mlt_alpha=1.8, convective_q_min=None,
        opacity_source="opal", kappa_es_factor=1.0,
        kappa_kramers_factor=1.0, kappa_hminus_factor=1.0,
        opal_X=0.70, opal_Z=0.02,
        opal_table=None,       # Phase D-0: OPAL テーブルのパス (None=自動探索。
                               # 旧版の絶対パス "/home/claude/GN93hz.dat" は他環境で
                               # 存在せず OPAL が黙って無効化されるバグの原因だった)
        opacity_lowT_guard="none",
        m_switch_frac=1e-6, center_smooth_points=3,
        ad_table_source="builtin-saha2d",
        saha_n_logT=25, saha_n_logrho=18,
        surface_boundary="opacity-source",
        surface_boundary_blend=0.0, surface_pressure_scale=1.0,
        max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        n_jac_workers=0,    # Jacobian 並列スレッド数 (0=CPU数を自動使用)
        n_sweep_workers=0,  # 質量掃引の並列プロセス数 (0=自動)
        transport="mlt-standard", schwarz_q_min=None,
        pp_rate_factor=1.0, cno_rate_factor=1.0, alpha_mlt_general=None,
        w_ml_prior=0.3,        # M-L 事前情報の重み (正しい分枝への誘導)
        w_ml_prior_scale=1.0,  # 段階的減衰用スケール (0=無効, 1=フル)
        w_energy_balance=2.0,  # Phase A-4: 大域エネルギー整合 ∫εdm=L (偽解排除)
        mesh_style='composite',# Phase B: 複合メッシュ (外層 ln(1-q) 均等)
        mesh_n_outer=40,       # 複合メッシュの外層点数
        tr_solver='lsmr',      # trf の部分問題ソルバー (lsmr: 大規模で高速)
        w_R_prior=2.0,         # R_star 事前情報の重み (Hayashi 偽解を抑制, tanh で有界)
        lowT_table=None,       # Phase D-1: F05 低温テーブルのパス (None=自動探索)
        lowT_blend_logT=4.0,   # Phase D-1: OPAL/F05 ブレンド中心 (logT)
        lowT_blend_dlogT=0.05, # Phase D-1: ブレンド半幅 (logT)
        mlt_solver='cubic',    # Phase D-2: 'cubic'=完全MLT / 'weak'=旧近似 (A/B用)
        L_zams_solar=0.72,     # ZAMS での太陽光度 (L☉ 単位)。M=1 の較正目標。
        R_zams_solar=0.87,     # ZAMS での太陽半径 (R☉ 単位)。参考値。
        plot=True,
    )






def write_table7(sol7, q, args, mode='s', outfile=None,
                 t_start=None, t_end=None, fit_dm=1e-4):
    """Phase C (fitting) 解のテーブル出力。write_table6 のフォーマットを
    そのまま使うが、以下を v7 規約に修正する:
      - R_solar / T_eff: v6 規約は接続点 (q_fit) を恒星表面と誤解釈するため、
        log_R_star とエンベロープ積分の T_eff で上書き
      - rad/c 列: choose_nabla7 (Saha EOS + 完全 MLT) でノード再評価
    テーブル範囲は q ≤ 1-fit_dm (それより外はエンベロープ積分領域)。"""
    N = len(q)
    class _FS:
        pass
    fs = _FS()
    fs.x = sol7.x[:-1]        # pack6 形式 (v6 互換)
    fs.fun = sol7.fun
    d = compute_derived6(fs, q, args)
    S, ell, p, t, lLe = unpack6(sol7.x[:-1], N)
    lR = float(sol7.x[-1])
    d['R_solar'] = float(np.exp(lR))
    d['R_star'] = float(np.exp(lR)) * R_sun
    env = integrate_envelope(np.exp(lR), np.exp(float(lLe)), args, fit_dm=fit_dm)
    if env is not None:
        d['T_eff'] = env['T_eff']
    # rad/c 判定を Phase C の物理 (Saha ∇_ad + MLT) でノード評価に置換
    _, nr7, na7 = choose_nabla7(q, S, p, ell, t, float(lLe), args)
    d['nabla_rad'] = nr7
    d['nabla_ad'] = na7
    write_table6(d, fs, args, t_start=t_start, t_end=t_end,
                 mode=mode, outfile=outfile)


def solve_star7_fitting(cli):
    """Phase C 一括パイプライン: メッシュ → 初期値 → EnvelopeTable →
    fitting ソルブ (粗箱 → 高精度小箱の 2 段) → 診断 → 保存 → テーブル出力"""
    import os as _os
    from datetime import datetime as _dt7
    _t_start7 = _dt7.now()
    global PP_CALIB, CNO_CALIB, M_hat
    apply_composition(cli.X, cli.Y, cli.Z)
    M_hat = float(cli.M)
    PP_CALIB  = 1.0 if cli.pp_calib  is None else float(cli.pp_calib)
    CNO_CALIB = 1.0 if cli.cno_calib is None else float(cli.cno_calib)
    args = build_default_args()
    args.mlt_alpha = cli.alpha_mlt
    args.lowT_table = getattr(cli, 'lowt_table', None)
    args.mlt_solver = getattr(cli, 'mlt_solver', 'cubic')
    args.w_ml_prior = 1.0
    args.w_R_prior = 3.0

    # ── Phase E: 非一様組成プロファイル (静的現在太陽モデル) ──
    _comp_path = getattr(cli, 'comp_profile', None)
    _present = getattr(cli, 'present_sun', False)
    if _comp_path is not None:
        prof = load_composition_profile(_comp_path)
        # グローバル組成を「表面」値に設定 (光球・対流層・参照定数の基準)。
        # 深部は _COMP_PROFILE 経由で自動的に非一様評価される。
        Xs = float(prof['X'][-1]); Zs = float(prof['Z'][-1])
        apply_composition(Xs, max(1.0 - Xs - Zs, 0.0), Zs)
        args.opal_X = Xs; args.opal_Z = Zs
        Xc = float(prof['X'][0])
        print(f"[Phase E] 組成プロファイル {_comp_path}: "
              f"表面 X={Xs:.4f} Z={Zs:.4f}, 中心 X={Xc:.4f} "
              f"(μ_surf={mu:.4f})")
        _present = True    # プロファイルを与えたら現在太陽モードを既定に
    else:
        set_composition_profile(None)   # 一様 (ZAMS)

    # ── prior 目標: ZAMS(既定) か 現在太陽(L=1,R=1) か ──
    if _present:
        args.L_zams_solar = 1.0    # 現在太陽 L=1 L☉
        args.R_zams_solar = 1.0    # 現在太陽 R=1 R☉
        print("[Phase E] 現在太陽モード: prior 目標 L=1.0 L☉, R=1.0 R☉")
    # prior 重みの明示上書き (prior 除去テスト用)
    if getattr(cli, 'w_ml_prior', None) is not None:
        args.w_ml_prior = float(cli.w_ml_prior)
    if getattr(cli, 'w_R_prior', None) is not None:
        args.w_R_prior = float(cli.w_R_prior)

    q = make_q_mesh_fit(cli.n_mesh, fit_dm=cli.fit_dm)
    N = len(q)
    print(f"[fitting] メッシュ N={N} (q_fit=1-{cli.fit_dm:g})  α_MLT={cli.alpha_mlt}")

    # ── 初期値: 保存解 (fit 解 → 通常解) の補間、なければ生成器 ──
    _dir = _os.path.dirname(_os.path.abspath(__file__))
    X0 = None
    for fn, is_fit in [("ph6_fit_sol.npz", True), ("ph6_solar_sol.npz", False)]:
        pth = _os.path.join(_dir, fn)
        if not _os.path.exists(pth):
            continue
        try:
            dd = np.load(pth, allow_pickle=True)
            Xo, qo = dd["X"], dd["q"]
            if is_fit and len(Xo) == 4 * len(qo) + 2:
                So, eo, po, to, lLe = unpack6(Xo[:-1], len(qo))
                lR = float(Xo[-1])
            elif not is_fit and len(Xo) == 4 * len(qo) + 1:
                So, eo, po, to, lLe = unpack6(Xo, len(qo))
                lR = float(So[-1])
            else:
                continue
            X0 = np.concatenate([pack6(
                np.interp(q, qo, So), np.interp(q, qo, eo),
                np.interp(q, qo, po), np.interp(q, qo, to),
                float(lLe)), [lR]])
            print(f"[fitting] 初期値: {fn} を補間 (N={len(qo)}→{N})")
            break
        except Exception as _e:
            print(f"[fitting] {fn} 読込失敗: {_e}")
    if X0 is None:
        # フォールバック: 通常メッシュで生成器 → 補間
        q_full = make_q_mesh(cli.n_mesh)
        S0, e0, p0, t0, lLe0 = make_initial_model6_envelope(q_full, M_hat, args)
        X0 = np.concatenate([pack6(
            np.interp(q, q_full, S0), np.interp(q, q_full, e0),
            np.interp(q, q_full, p0), np.interp(q, q_full, t0),
            lLe0), [float(S0[-1])]])
        print("[fitting] 初期値: ポリトロープ+ブリッジ生成器から補間")

    lo6, hi6 = make_bounds6(N)
    lo = np.concatenate([lo6, [np.log(0.3)]])
    hi = np.concatenate([hi6, [np.log(3.0)]])
    X0 = np.clip(X0, lo, hi)

    from scipy.optimize import least_squares as _lsq
    import time as _time

    # ── 第 1 段: 粗箱テーブル ──
    t0 = _time.time()
    tab = EnvelopeTable(args, fit_dm=cli.fit_dm, n=4, half_R=0.04, half_L=0.12)
    if not tab._build(float(X0[-1]), float(X0[4 * N])):
        print("[fitting] エンベロープ積分に失敗 (初期 R, L を確認)")
        return None
    print(f"[fitting] EnvelopeTable (粗) {_time.time()-t0:.0f}s")
    _rf = lambda Xv, qq, aa: residual_vector7(Xv, qq, aa, tab)
    t0 = _time.time()
    sol = _lsq(_rf, X0, args=(q, args), bounds=(lo, hi), method="trf",
               jac="2-point", tr_solver="lsmr", x_scale="jac",
               ftol=1e-12, xtol=1e-12, gtol=1e-12,
               max_nfev=cli.fit_nfev, verbose=0)
    print(f"[fitting] 第1段 {_time.time()-t0:.0f}s nfev={sol.nfev}: "
          f"|res|∞={np.max(np.abs(sol.fun)):.3e}")

    # ── 第1段チェックポイント: 中断されても結果を失わないよう即保存 ──
    try:
        _ckpt = _os.path.join(_dir, "ph6_fit_sol.npz")
        np.savez(_ckpt, X=sol.x, q=q, M_hat=M_hat, stage=1)
        print(f"[fitting] 第1段チェックポイント保存: {_ckpt}")
    except Exception as _e:
        print(f"[fitting] チェックポイント保存失敗: {_e}")

    # ── 第 2 段: 解の周りの高精度小箱で磨く ──
    t0 = _time.time()
    tab2 = EnvelopeTable(args, fit_dm=cli.fit_dm, n=4,
                         half_R=0.012, half_L=0.035)
    if tab2._build(float(sol.x[-1]), float(sol.x[4 * N])):
        _rf2 = lambda Xv, qq, aa: residual_vector7(Xv, qq, aa, tab2)
        sol = _lsq(_rf2, sol.x, args=(q, args), bounds=(lo, hi), method="trf",
                   jac="2-point", tr_solver="lsmr", x_scale="jac",
                   ftol=1e-13, xtol=1e-13, gtol=1e-13,
                   max_nfev=80, verbose=0)
        print(f"[fitting] 第2段 {_time.time()-t0:.0f}s: "
              f"|res|∞={np.max(np.abs(sol.fun)):.3e}")

    # ── 診断 ──
    S, ell, p, t, lLe = unpack6(sol.x[:-1], N)
    lR = float(sol.x[-1]); lL = float(lLe)
    T_c = T_REF * np.exp(t[0]); P_c = P_REF * np.exp(p[0])
    _Xc7, _Yc7, _Zc7, _muc7, _XCNOc7 = _comp_at(np.array([q[0]]))
    rho_c = float(density_from_PT(np.array([P_c]), np.array([T_c]),
                                  mu_loc=_muc7)[0])
    env = integrate_envelope(np.exp(lR), np.exp(lL), args, fit_dm=cli.fit_dm)
    _Xd, _Yd, _Zd, _mud, _XCNOd = _comp_at(q)   # Phase E: 診断も非一様組成
    P_n = P_REF * np.exp(p); T_n = T_REF * np.exp(t)
    eps_n = energy_generation(density_from_PT(P_n, T_n, mu_loc=_mud), T_n,
                              X_loc=_Xd, XCNO_loc=_XCNOd)
    L_mass = float(np.sum(0.5 * (eps_n[:-1] + eps_n[1:]) * np.diff(q))) * M_hat * M_sun
    L_star = L_sun * np.exp(lL) * float(ell[-1])
    print("═" * 60)
    print(f"[fitting] R={np.exp(lR):.4f} R☉  L={L_star/L_sun:.4f} L☉  "
          f"T_eff={env['T_eff']:.0f} K" if env else "")
    print(f"[fitting] T_c={T_c/1e6:.2f} MK  ρ_c={rho_c:.1f} g/cm³  "
          f"∫εdm/L={L_mass/max(L_star,1e-9):.4f}")
    qm = 0.5 * (q[:-1] + q[1:])
    nab, nrv, nav = choose_nabla7(qm, 0.5*(S[:-1]+S[1:]), 0.5*(p[:-1]+p[1:]),
                                  0.5*(ell[:-1]+ell[1:]), 0.5*(t[:-1]+t[1:]),
                                  lL, args)
    print(f"[fitting] 中心: ∇_rad={nrv[0]:.3f} / ∇_ad={nav[0]:.3f} "
          f"({'対流コア' if nrv[0]>nav[0] else '放射的'})")
    kk = np.where((nrv > nav) & (qm > 0.5))[0]
    if len(kk):
        k0 = kk[0]
        print(f"[fitting] 表面対流層底: q={qm[k0]:.5f}  "
              f"r={np.exp(0.5*(S[k0]+S[k0+1])):.4f} R☉  "
              f"T={T_REF*np.exp(0.5*(t[k0]+t[k0+1])):.3e} K "
              f"(以深はエンベロープまで対流)")
    # ── 保存 ──
    _save = _os.path.join(_dir, "ph6_fit_sol.npz")
    np.savez(_save, X=sol.x, q=q, PP_CALIB=PP_CALIB,
             R_solar=float(np.exp(lR)), L_solar=float(L_star / L_sun),
             T_c=float(T_c), rho_c=float(rho_c),
             res_inf=float(np.max(np.abs(sol.fun))), n_mesh=N,
             fit_dm=cli.fit_dm, mlt_alpha=cli.alpha_mlt)
    print(f"[fitting] 解を保存: {_save}")

    # ── テーブル出力 (--table-out s|l) ──
    _t_end7 = _dt7.now()
    if getattr(cli, 'table_out', 'n') != 'n':
        tfile = f"{cli.out_prefix}_M{cli.M:.2f}_table.txt"
        write_table7(sol, q, args, mode=cli.table_out, outfile=tfile,
                     t_start=_t_start7, t_end=_t_end7, fit_dm=cli.fit_dm)
    return sol


def main():
    from datetime import datetime as _dt

    pa = argparse.ArgumentParser(description="Phase 6: 真の ZAMS")
    pa.add_argument("--M",          type=float, default=1.0)
    pa.add_argument("--X",          type=float, default=0.70)
    pa.add_argument("--Y",          type=float, default=0.28)
    pa.add_argument("--Z",          type=float, default=0.02)
    pa.add_argument("--n-mesh",     type=int,   default=80)
    pa.add_argument("--max-nfev",   type=int,   default=2000)
    pa.add_argument("--jac-workers",   type=int, default=0,
                    help="Jacobian 並列スレッド数 (0=CPU数自動, 1=直列)")
    pa.add_argument("--sweep-workers", type=int, default=0,
                    help="質量掃引の並列プロセス数 (0=CPU数自動, 1=直列)")
    pa.add_argument("--alpha-mlt",  type=float, default=1.8)
    pa.add_argument("--lowt-table", type=str,   default=None,
                    help="Phase D-1: F05 低温 opacity テーブルのパス "
                         "(デフォルト: スクリプトと同じ場所の F05_lowT_g98.dat)")
    pa.add_argument("--mlt-solver", type=str,   default="cubic",
                    choices=["cubic", "weak"],
                    help="Phase D-2: MLT ソルバー (cubic=完全MLT[標準], weak=旧近似)")
    pa.add_argument("--fitting",    action="store_true",
                    help="Phase C: fitting point 法 + 完全 MLT + Saha EOS で解く")
    pa.add_argument("--comp-profile", type=str, default=None,
                    help="Phase E: 非一様組成 X(q) プロファイル (列: q X Z)。"
                         "指定すると静的な現在太陽モデルを解く。例: modelS_Xq.dat")
    pa.add_argument("--present-sun", action="store_true",
                    help="Phase E: 現在太陽モード. prior 目標を L=1, R=1 に設定"
                         " (ZAMS の L=0.72, R=0.87 ではなく)")
    pa.add_argument("--w-ml-prior", type=float, default=None,
                    help="M-L prior の重み上書き (prior 除去テスト用, 0 で無効)")
    pa.add_argument("--w-R-prior",  type=float, default=None,
                    help="R prior の重み上書き (prior 除去テスト用, 0 で無効)")
    pa.add_argument("--fit-dm",     type=float, default=1e-4,
                    help="接続点の外側質量 1-q_fit (default 1e-4)")
    pa.add_argument("--fit-nfev",   type=int,   default=400)
    pa.add_argument("--calibrate",  action="store_true")
    pa.add_argument("--ph4-init",   type=str,   default=None)
    pa.add_argument("--sweep",      type=float, nargs="+", default=None)
    pa.add_argument("--pp-calib",       type=float, default=None)
    pa.add_argument("--cno-calib",      type=float, default=None)
    pa.add_argument("--pp-calib-auto",  action="store_true",
                    help="PP_CALIB を L_total(∫ε)から自動推定して初期設定する")
    pa.add_argument("--smooth-eps",     type=float, default=0.03,
                    help="Schwarzschild 判定の tanh 平滑化幅 (デフォルト: 0.03)")
    pa.add_argument("--no-plot",    action="store_true")
    pa.add_argument("--out-prefix", type=str,   default="ph6")
    pa.add_argument(
        "--table-out",
        type=str, default="n",
        choices=["n", "s", "l"],
        metavar="[n|s|l]",
        help=(
            "テキストテーブル出力モード: "
            "n=なし(デフォルト), "
            "s=small(20-25行に間引き), "
            "l=large(全メッシュ)"
        ),
    )
    cli = pa.parse_args()

    if getattr(cli, 'fitting', False):
        solve_star7_fitting(cli)
        return

    # ── 計算開始時刻 ──────────────────────────────────
    t_start = _dt.now()
    print(f"計算開始: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    _auto_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'ph6_solar_sol.npz')
    print(f"  解の自動保存先: {_auto_save_path}")

    global PP_CALIB, CNO_CALIB
    args = build_default_args()
    args.n_mesh=cli.n_mesh; args.max_nfev=cli.max_nfev; args.mlt_alpha=cli.alpha_mlt
    args.lowT_table=getattr(cli,'lowt_table',None); args.mlt_solver=getattr(cli,'mlt_solver','cubic')
    args.X_init=cli.X; args.Y_init=cli.Y; args.Z_init=cli.Z
    args.opal_X=cli.X; args.opal_Z=cli.Z; args.plot=not cli.no_plot
    args.schwarz_smooth_eps = cli.smooth_eps   # tanh 平滑化幅
    args.n_jac_workers   = cli.jac_workers     # Jacobian 並列スレッド数
    args.n_sweep_workers = cli.sweep_workers   # 質量掃引 並列プロセス数

    if _HAVE_PH4:
        _ph4.X=cli.X; _ph4.Y=cli.Y; _ph4.Z=cli.Z
        _ph4.mu=1.0/max(2*cli.X+0.75*cli.Y+0.5*cli.Z,1e-99)

    if cli.pp_calib  is not None: PP_CALIB  = cli.pp_calib
    if cli.cno_calib is not None: CNO_CALIB = cli.cno_calib

    # ── PP_CALIB 自動一次較正 ─────────────────────────────────────
    # --pp-calib が明示指定されていない場合、Lane-Emden 積分で L_total を推定し
    # PP_CALIB ≈ L_zams_target / L_total(PP=1) で初期較正する。
    # --pp-calib 明示指定の場合はスキップ。
    # ── Phase A: 較正係数は原則 1.0 ─────────────────────────────────
    # エネルギー生成率が Kippenhahn & Weigert 標準式になったため、
    # Lane-Emden ポリトロープによる自動較正 (旧 PP≈0.39) は廃止した。
    # 標準式のまま (PP_CALIB=1) で太陽 ZAMS の L≈0.72 L☉ が再現されることが
    # 物理実装の検証条件となる。--pp-calib での明示指定は引き続き可能。
    if cli.pp_calib is None:
        print(f"  PP_CALIB=1.0 (標準レート式, Lane-Emden 自動較正は廃止)")

    start_sol=None; start_q=None
    if cli.calibrate:
        sol_c,q_c,d_c = calibrate_solar6(args, ph4_init=cli.ph4_init)
        print(f"\n  較正後: PP={PP_CALIB:.4f}  CNO={CNO_CALIB:.4f}")
        start_sol,start_q = sol_c, q_c

    table_mode = cli.table_out  # 'n', 's', 'l'

    if cli.sweep:
        results = mass_sweep6(cli.sweep, args, start_sol, start_q)

        # ── 計算終了時刻 ──────────────────────────────
        t_end = _dt.now()
        elapsed = (t_end - t_start).total_seconds()

        print("\n" + "="*72 + "\n  掃引結果")
        print(f"  計算開始: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  計算終了: {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  所要時間: {elapsed:.3f} 秒")
        print(f"  {'M':>8} {'R':>8} {'L':>12} {'T_c':>9} {'T_eff':>9} {'CNO':>7}")
        print("-"*72)
        for M,sol,q,d in sorted(results, key=lambda x: x[0]):
            print(f"  {M:>8.3f} {d['R_solar']:>8.4f} {d['L_solar']:>12.4e} "
                  f"{d['T_c']/1e6:>9.3f} {d['T_eff']:>9.1f} {d['cno_frac']:>7.4f}")

        if args.plot:
            plot_sweep6(results, f"{cli.out_prefix}_sweep.png")
            for M,sol,q,d in results:
                plot_solution6(sol,q,d,args, f"{cli.out_prefix}_M{M:.2f}.png")

        # テーブル出力（掃引時は各 M ごとにファイル出力）
        if table_mode != 'n':
            for M,sol,q,d in results:
                tfile = f"{cli.out_prefix}_M{M:.2f}_table.txt"
                write_table6(d, sol, args,
                             t_start=t_start, t_end=t_end,
                             mode=table_mode, outfile=tfile)
    else:
        if cli.ph4_init and os.path.exists(cli.ph4_init):
            apply_composition(cli.X, cli.Y, cli.Z)
            global M_hat; M_hat=cli.M
            q_tmp = make_q_mesh(cli.n_mesh)
            S0,ell0,p0,t0,lLe0 = load_ph4_solution_as_ph6_init(cli.ph4_init, q_tmp, args)
            X0ph4 = pack6(S0,ell0,p0,t0,lLe0)
            sol,q,d = solve_star6(cli.M, args, X0=X0ph4, q_given=q_tmp)
        else:
            sol,q,d = solve_star6(cli.M, args)

        # ── 計算終了時刻 ──────────────────────────────
        t_end = _dt.now()
        elapsed = (t_end - t_start).total_seconds()

        print_summary6(sol, d, args, t_start=t_start, t_end=t_end)

        if args.plot:
            plot_solution6(sol,q,d,args, f"{cli.out_prefix}_M{cli.M:.2f}.png")

        # テーブル出力
        if table_mode != 'n':
            tfile = f"{cli.out_prefix}_M{cli.M:.2f}_table.txt"
            write_table6(d, sol, args,
                         t_start=t_start, t_end=t_end,
                         mode=table_mode, outfile=tfile)

    print(f"\n計算終了: {t_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所要時間: {elapsed:.3f} 秒")
    print("\n完了。")




# ═══════════════════════════════════════════════════════════════════════
#  Phase C: Saha 電離 EOS + 完全 MLT + fitting point 法
#
#  背景: Phase A/B 後も表面対流層が形成されなかった根因は 2 つ:
#   (1) EOS の ∇_ad に H/He 電離の熱力学がなく (常に ≈0.4)、電離帯で
#       温度上昇が抑制されない → エントロピー過大 → κ 低下 → 早期放射化
#   (2) 効率対流近似 (∇=∇_ad) では光球直下の超断熱層を表現できず、
#       対流層断熱線のエントロピーが決まらない
#  対策: Saha EOS (μ, δ, c_P, ∇_ad) + K&W §7 の完全 MLT 3次方程式 +
#  光球〜q_fit の内向きエンベロープ積分 (fitting point 法)。
#  超断熱層・電離帯 (1-q < fit_dm) は Henyey 緩和から切り離す。
# ═══════════════════════════════════════════════════════════════════════
_m_e_C  = 9.1093837e-28
_h_pl_C = 6.62607015e-27
_eV_C   = 1.602176634e-12
_CHI_H   = 13.598 * _eV_C
_CHI_HE1 = 24.587 * _eV_C
_CHI_HE2 = 54.418 * _eV_C


def saha_state(rho, T):
    """(ρ,T) → x_H+, f_HeII, f_HeIII, μ, u [erg/g]  (H + He の Saha 電離)"""
    kT = k_B * T
    nH = X * rho / m_H
    nHe = Y * rho / (4.0 * m_H)
    lam = (2.0 * np.pi * _m_e_C * kT / _h_pl_C**2) ** 1.5
    KH = lam * np.exp(-min(_CHI_H / kT, 500.0))
    K1 = 4.0 * lam * np.exp(-min(_CHI_HE1 / kT, 500.0))
    K2 = lam * np.exp(-min(_CHI_HE2 / kT, 500.0))
    ne = 0.5 * (nH + 2.0 * nHe)
    for _ in range(60):
        xh = KH / (KH + ne)
        dd = 1.0 + K1 / ne + K1 * K2 / ne**2
        f2 = (K1 / ne) / dd
        f3 = (K1 * K2 / ne**2) / dd
        ne_new = xh * nH + (f2 + 2.0 * f3) * nHe + 1e-8 * nH
        if abs(ne_new - ne) < 1e-10 * ne:
            ne = ne_new
            break
        ne = 0.5 * (ne + ne_new)
    xh = KH / (KH + ne)
    dd = 1.0 + K1 / ne + K1 * K2 / ne**2
    f2 = (K1 / ne) / dd
    f3 = (K1 * K2 / ne**2) / dd
    inv_mu = X * (1.0 + xh) + Y * (1.0 + f2 + 2.0 * f3) / 4.0 + Z / 2.0
    mu_v = 1.0 / inv_mu
    n_tot = rho / (mu_v * m_H)
    u = (1.5 * kT * n_tot / rho
         + (_CHI_H * xh * nH
            + (_CHI_HE1 * f2 + (_CHI_HE1 + _CHI_HE2) * f3) * nHe) / rho
         + a_rad * T**4 / rho)
    return xh, f2, f3, mu_v, u


def saha_rho(P, T):
    """(P,T) → ρ (Saha μ を自己無撞着に解く)"""
    Pg = max(P - a_rad * T**4 / 3.0, 1e-3 * P)
    mu_v = 0.617
    for _ in range(40):
        rho = Pg * mu_v * m_H / (k_B * T)
        _, _, _, mu_new, _ = saha_state(rho, T)
        if abs(mu_new - mu_v) < 1e-10 * mu_v:
            mu_v = mu_new
            break
        mu_v = 0.5 * (mu_v + mu_new)
    return Pg * mu_v * m_H / (k_B * T)


from functools import lru_cache as _lru_C

@_lru_C(maxsize=400000)
def saha_thermo(P, T):
    """(P,T) → ρ, δ, c_P, ∇_ad  (有限差分, 電離込み)
    δ = -(∂lnρ/∂lnT)_P,  c_P = (∂h/∂T)_P,  ∇_ad = Pδ/(Tρc_P) (K&W 4.21)
    注: FD Jacobian では大半の中点の (P,T) が呼び出し間でビット単位に
    一致するため lru_cache が非常に効く (残差評価の実効コストが激減)。"""
    eps_fd = 3e-4
    rho = saha_rho(P, T)
    rp = saha_rho(P, T * np.exp(eps_fd))
    rm = saha_rho(P, T * np.exp(-eps_fd))
    delta = max(-(np.log(rp) - np.log(rm)) / (2.0 * eps_fd), 1e-3)
    _, _, _, _, up = saha_state(rp, T * np.exp(eps_fd))
    _, _, _, _, um = saha_state(rm, T * np.exp(-eps_fd))
    cP = ((up + P / rp) - (um + P / rm)) / (2.0 * eps_fd * T)
    cP = max(cP, 1e-30)
    na = P * delta / (T * rho * cP)
    return rho, delta, cP, float(np.clip(na, 0.02, 0.5))


def mlt_nabla_full(rho, T, P, kap, g, nr, na, delta, cP, alpha):
    """K&W §7 の完全 MLT (スカラー版)。
    Phase D-2: 共通ソルバー _mlt_cubic_eta (ベクトル化二分法) に委譲。
    旧実装の brentq と <1e-12 で一致することを verify_phase_d.py で確認済み。"""
    W = nr - na
    if W <= 0.0:
        return nr
    H_P = P / (rho * g)
    ell_m = alpha * H_P
    U = (3.0 * a_rad * c_light * T**3) / (cP * rho**2 * kap * ell_m**2) \
        * np.sqrt(8.0 * H_P / (g * delta))
    eta = float(_mlt_cubic_eta(U, W)[0])
    return na + eta * eta + 2.0 * U * eta


def integrate_envelope(R_hat, L_hat, args, fit_dm=1e-4):
    """光球 (τ=2/3, Saha 自己無撞着) から 1-q=fit_dm まで完全 MLT で
    内向き積分。独立変数 x=lnP、状態 y=[lnr, μ(外側質量), lnT]。
    Returns dict(r_fit, P_fit, T_fit, T_eff, P_phot) / None (失敗時)"""
    from scipy.integrate import solve_ivp as _ivp
    R_star = R_hat * R_sun
    L_star = L_hat * L_sun
    M_star = M_hat * M_sun
    T_eff = (L_star / (4.0 * np.pi * R_star**2 * (a_rad * c_light / 4.0)))**0.25
    g_s = G * M_star / R_star**2
    P0 = 1e4
    for _ in range(8):
        rho0 = saha_rho(P0, T_eff)
        kap0 = float(opacity(np.array([rho0]), np.array([T_eff]), args)[0])
        P0 = 2.0 * g_s / (3.0 * max(kap0, 1e-8))
    mu0 = 4.0 * np.pi * R_star**4 * P0 / (G * M_star)
    mu_t = fit_dm * M_star
    alpha = getattr(args, 'mlt_alpha', 1.8)

    def _rhs(x, y):
        lnr, mu_m, lnT = y
        P = np.exp(x); r = np.exp(lnr); T = np.exp(lnT)
        m = M_star - mu_m
        rho, delta, cP, na = saha_thermo(P, T)
        kap = float(opacity(np.array([rho]), np.array([T]), args)[0])
        g = G * m / r**2
        nr = 3.0 * kap * L_star * P / (16.0 * np.pi * a_rad * c_light * G * m * T**4)
        nab = mlt_nabla_full(rho, T, P, kap, g, nr, na, delta, cP, alpha)
        return [-r * P / (G * m * rho),
                4.0 * np.pi * r**4 * P / (G * m),
                nab]

    ev = lambda x, y: y[1] - mu_t
    ev.terminal = True
    ev.direction = 1.0
    try:
        sol = _ivp(_rhs, [np.log(P0), np.log(P0) + 45.0],
                   [np.log(R_star), mu0, np.log(T_eff)],
                   method='RK45', rtol=1e-6,
                   atol=[1e-7, mu_t * 1e-9, 1e-7],
                   events=ev, max_step=0.5)
    except Exception:
        return None
    if not sol.t_events[0].size:
        return None
    xe = sol.t_events[0][0]
    ye = sol.y_events[0][0]
    return dict(r_fit=float(np.exp(ye[0])), P_fit=float(np.exp(xe)),
                T_fit=float(np.exp(ye[2])), T_eff=float(T_eff),
                P_phot=float(P0))


class EnvelopeTable:
    """(lnR, lnL) の小格子上で integrate_envelope を事前計算し双線形補間。
    残差評価のたびの ODE 積分 (~0.5-2s) を回避する。格子外に出たら再構築。"""
    def __init__(self, args, fit_dm=1e-4, n=4, half_R=0.04, half_L=0.12):
        self.args = args; self.fit_dm = fit_dm
        self.n = n; self.hR = half_R; self.hL = half_L
        self.box = None

    def _build(self, lnR0, lnL0):
        n = self.n
        self.gR = np.linspace(lnR0 - self.hR, lnR0 + self.hR, n)
        self.gL = np.linspace(lnL0 - self.hL, lnL0 + self.hL, n)
        self.tab = np.zeros((n, n, 3))
        for i, lr in enumerate(self.gR):
            for j, ll in enumerate(self.gL):
                env = integrate_envelope(np.exp(lr), np.exp(ll),
                                         self.args, self.fit_dm)
                if env is None:
                    return False
                self.tab[i, j] = [np.log(env['r_fit'] / R_sun),
                                  np.log(env['P_fit'] / P_REF),
                                  np.log(env['T_fit'] / T_REF)]
        self.box = (lnR0, lnL0)
        return True

    def query(self, lnR, lnL):
        """→ (S_fit, p_fit, t_fit) / None"""
        if (self.box is None
                or abs(lnR - self.box[0]) > self.hR
                or abs(lnL - self.box[1]) > self.hL):
            if not self._build(lnR, lnL):
                return None
        iR = int(np.clip(np.searchsorted(self.gR, lnR) - 1, 0, self.n - 2))
        iL = int(np.clip(np.searchsorted(self.gL, lnL) - 1, 0, self.n - 2))
        fR = (lnR - self.gR[iR]) / (self.gR[iR + 1] - self.gR[iR])
        fL = (lnL - self.gL[iL]) / (self.gL[iL + 1] - self.gL[iL])
        fR = np.clip(fR, 0.0, 1.0); fL = np.clip(fL, 0.0, 1.0)
        v = ((1 - fR) * (1 - fL) * self.tab[iR, iL]
             + fR * (1 - fL) * self.tab[iR + 1, iL]
             + (1 - fR) * fL * self.tab[iR, iL + 1]
             + fR * fL * self.tab[iR + 1, iL + 1])
        return v[0], v[1], v[2]


def choose_nabla7(qmid, Smid, pmid, emid, tmid, log_L_est, args):
    """Phase C 内部用 ∇: 完全 MLT。
    T < 1e6 K では Saha 熱力学 (∇_ad の電離 dip)、高温側は既存 nabla_ad。
    fitting point (1-q=1e-4, T≈2×10⁵) 以深が対象なので Saha 分岐は
    接続点付近の数点のみ → コスト増は限定的。"""
    r = R_sun * np.exp(np.clip(Smid, -80, 20))
    m = np.maximum(qmid, 1e-12) * M_hat * M_sun
    P = P_REF * np.exp(np.clip(pmid, -200, 200))
    T = T_REF * np.exp(np.clip(tmid, -100, 50))
    X_q, Y_q, Z_q, mu_q, XCNO_q = _comp_at(qmid)     # Phase E: 非一様組成
    rho = density_from_PT(P, T, mu_loc=mu_q)
    kap = opacity(rho, T, args, X_loc=X_q, Z_loc=Z_q)
    L = L_sun * np.exp(float(log_L_est)) * np.maximum(emid, 0.0)
    g = G * m / r**2
    # 中心近傍では L/m の離散値のノイズ (ell の残差レベルの揺らぎ×1/q) が
    # 増幅されるため、Phase A-3 と同じ解析極限 L/m → ε(ρ,T) を使う
    # (L(m)=∫εdm ≈ ε m; q≲3e-3 では両者は ~1% 以内で一致)。
    Lom = L / m
    _cen = qmid < 3e-3
    if np.any(_cen):
        eps_mid = energy_generation(rho, T, X_loc=X_q, XCNO_loc=XCNO_q)
        Lom = np.where(_cen, eps_mid, Lom)
    nr = 3.0 * kap * Lom * P / (16.0 * np.pi * a_rad * c_light * G * T**4)
    nr = np.clip(nr, 0.0, 1e6)
    alpha = getattr(args, 'mlt_alpha', 1.8)
    T_sw = getattr(args, 'saha_T_switch', 1.0e6)
    # 高温側 (完全電離) は既存 nabla_ad を全点一括評価
    na = np.asarray(nabla_ad(rho, T, P, args, mu_loc=mu_q), dtype=float).copy()
    delta_v = np.ones_like(na)
    cP_v = P / (T * rho * np.maximum(na, 1e-3))
    rho_v = rho.astype(float).copy()
    # 低温側 (Saha) のみ点ごとに上書き
    cold = np.where(T < T_sw)[0]
    for i in cold:
        rho_i, delta_i, cP_i, na_i = saha_thermo(float(P[i]), float(T[i]))
        na[i] = na_i; delta_v[i] = delta_i; cP_v[i] = cP_i; rho_v[i] = rho_i
    # MLT は対流点 (nr > na) のみ — Phase D-2: ベクトル化した完全 MLT
    nab = np.minimum(nr, na)  # 効率極限を仮置き
    conv = nr > na
    if np.any(conv):
        H_P_c   = P[conv] / (rho_v[conv] * g[conv])
        ell_c   = alpha * H_P_c
        U_c = (3.0 * a_rad * c_light * T[conv]**3
               / (cP_v[conv] * rho_v[conv]**2 * kap[conv] * ell_c**2)
               * np.sqrt(8.0 * H_P_c / (g[conv] * delta_v[conv])))
        eta_c = _mlt_cubic_eta(U_c, nr[conv] - na[conv])
        nab[conv] = na[conv] + eta_c * (eta_c + 2.0 * U_c)
    rad_idx = np.where(nr <= na)[0]
    nab[rad_idx] = nr[rad_idx]
    return nab, nr, na


def make_q_mesh_fit(N=80, fit_dm=1e-4, n_outer=16, q0=1e-8):
    """fitting point 法用の内部メッシュ: [q0, 1-fit_dm]。
    外層は ln(1-q) 均等で 1-q=fit_dm まで (光球側は緩和から除外)。"""
    q = make_q_mesh(N, q0=q0, mesh_style='composite',
                    n_outer=n_outer, outer_dm_min=fit_dm)
    return q[q <= 1.0 - fit_dm + 1e-12]


def residual_vector7(X, q, args, env_table):
    """Phase C 残差 (fitting point 法 + 完全 MLT)。
    未知数: [pack6(S,ell,p,t,log_L_est), log_R_star] の 4N+2 個。
    表面 BC (光球) の代わりに接続点 4 条件:
      S[-1]=ln r_env, p[-1]=ln P_env, t[-1]=ln T_env, ell[-1]=1
    をエンベロープ積分 (EnvelopeTable 補間) から課す。"""
    N = len(q)
    log_R_star = float(X[-1])
    S, ell, p, t, log_L_est = unpack6(X[:-1], N)

    dq = np.diff(q)
    qmid = 0.5 * (q[:-1] + q[1:])
    Smid = 0.5 * (S[:-1] + S[1:])
    emid = 0.5 * (ell[:-1] + ell[1:])
    pmid = 0.5 * (p[:-1] + p[1:])
    tmid = 0.5 * (t[:-1] + t[1:])

    r, m, P, L, T, rho = physical6(qmid, Smid, pmid, emid, tmid, log_L_est)
    nabla, nr, na = choose_nabla7(qmid, Smid, pmid, emid, tmid, log_L_est, args)

    # ノード ε (Rell / R_energy 共通, Phase A-4)
    Xn7, Yn7, Zn7, mun7, XCNOn7 = _comp_at(q)     # Phase E
    P_n = P_REF * np.exp(np.clip(p, -200, 200))
    T_n = T_REF * np.exp(np.clip(t, -100, 50))
    rho_n = density_from_PT(P_n, T_n, mu_loc=mun7)
    eps_n = energy_generation(rho_n, T_n, X_loc=Xn7, XCNO_loc=XCNOn7)
    eps_tz = 0.5 * (eps_n[:-1] + eps_n[1:])

    # ── Rs (v6 と同じ hybrid スケーリング) ──
    Rs_model = dq * M_hat * RHO_S / (rho * np.exp(3 * np.clip(Smid, -80, 20)))
    Rs_raw = (S[1:] - S[:-1]) - Rs_model
    floor_rs = getattr(args, 'rs_floor', 1e-8)
    w_rel = getattr(args, 'rs_relative_weight', 1.0)
    w_abs = getattr(args, 'outer_rs_raw_weight', 1.0)
    sw = sigmoid((qmid - getattr(args, 'outer_q_start', 0.5)) /
                 max(getattr(args, 'outer_q_width', 0.1), 1e-12))
    denom = np.abs(S[1:] - S[:-1]) + np.abs(Rs_model) + floor_rs
    Rs = (1 - sw) * w_rel * Rs_raw / denom + sw * w_abs * Rs_raw

    # ── Rp / Rell / Rt ──
    Rp_model = -dq * M_hat**2 * qmid / np.exp(np.clip(pmid + 4 * Smid, -200, 200))
    Rp = (p[1:] - p[:-1]) - Rp_model
    Rell = (ell[1:] - ell[:-1]) - dq * M_hat * eps_tz / (EPS_REF * np.exp(float(log_L_est)))
    Rt = (t[1:] - t[:-1]) - nabla * Rp_model

    # ── 接続点条件 (エンベロープ) ──
    envq = env_table.query(log_R_star, float(log_L_est))
    if envq is None:
        Rbc_S = Rbc_p = Rbc_t = 10.0
    else:
        S_fit, p_fit, t_fit = envq
        Rbc_S = S[-1] - S_fit
        Rbc_p = p[-1] - p_fit
        Rbc_t = t[-1] - t_fit
    Rbc_ell = ell[-1] - 1.0

    # ── 中心境界条件 (v6 と同一) ──
    _r0, _m0, P0, _L0, T0, rho0 = physical6(
        np.array([q[0]]), np.array([S[0]]),
        np.array([p[0]]), np.array([ell[0]]), np.array([t[0]]), log_L_est)
    rho_c = float(rho0[0]); T_c = float(T0[0])
    _Xc, _Yc, _Zc, _muc, _XCNOc = _comp_at(np.array([q[0]]))
    eps_c = float(energy_generation(np.array([rho_c]), np.array([T_c]),
                                    X_loc=_Xc, XCNO_loc=_XCNOc)[0])
    S0_exp = (1.0 / 3.0) * np.log(max(3 * M_hat * q[0] / (4 * np.pi * rho_c * RHO_S), 1e-200))
    L_est = L_sun * np.exp(float(log_L_est))
    ell0_exp = eps_c * M_hat * M_sun * q[0] / max(L_est, 1e-99)
    Rcen_S = S[0] - S0_exp
    Rcen_ell = ell[0] - ell0_exp
    w_cT = getattr(args, 'weight_center_T', 0.0)
    w_crho = getattr(args, 'weight_center_rho', 0.0)
    Rcen_T = w_cT * (t[0] - np.log(max(T_c / T_REF, 1e-99)))
    Rcen_rho = w_crho * 0.0

    # ── prior (M-L / R) ──
    w_ml = getattr(args, 'w_ml_prior', 0.0) * getattr(args, 'w_ml_prior_scale', 1.0)
    if w_ml > 0:
        log_off = np.log(max(getattr(args, 'L_zams_solar', 0.72), 1e-10))
        log_L_exp = (4.0 if M_hat >= 0.7 else 4.5) * np.log(max(M_hat, 0.1)) + log_off
        Rcen_ML = w_ml * (log_L_est - log_L_exp)
    else:
        Rcen_ML = 0.0
    w_R = getattr(args, 'w_R_prior', 2.0)
    if w_R > 0.0:
        S_target = np.log(max(M_hat, 0.1)**0.8 * getattr(args, 'R_zams_solar', 0.87))
        Rcen_R = w_R * np.tanh((log_R_star - S_target) / 0.5)
    else:
        Rcen_R = 0.0

    # ── 大域エネルギー整合 (Phase A-4) ──
    w_E = getattr(args, 'w_energy_balance', 2.0)
    if w_E > 0:
        L_mass_v = float(np.sum(eps_tz * dq)) * M_hat * M_sun
        L_star_v = L_sun * np.exp(float(log_L_est)) * max(float(ell[-1]), 1e-30)
        R_energy = w_E * float(np.clip(
            np.log(max(L_mass_v, 1e-30) / max(L_star_v, 1e-30)) / 2.0, -3.0, 3.0))
    else:
        R_energy = 0.0

    w_s = getattr(args, 'weight_structure', 1.0)
    w_b = getattr(args, 'weight_bc', 1.0)
    return np.concatenate([
        w_s * Rs, w_s * Rp, w_s * Rell, w_s * Rt,
        [w_b * Rbc_t, w_b * Rbc_p, w_b * Rbc_S, w_b * Rbc_ell],
        [Rcen_S, Rcen_ell, Rcen_T, Rcen_rho, Rcen_ML, Rcen_R, R_energy],
    ])


if __name__ == "__main__":
    main()
