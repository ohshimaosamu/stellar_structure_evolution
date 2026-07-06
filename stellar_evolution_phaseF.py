#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stellar_evolution_phaseF.py — Phase F: 主系列進化 + 太陽較正
==============================================================

Phase E で「文献 X(q) を静的ソルバーに与えれば現在の太陽が再現される」ことを
確認した。Phase F はその X(q) を自前で生成する:

  (a) 核燃焼による組成発展 (operator splitting):
      各タイムステップで
        1. 現在の X(q) で静的構造を解く (stellar_structure_phase6-10 を再利用)
        2. ε(q) から dX/dt = -(ε_pp/E_pp + ε_CNO/E_CNO) で X を更新
        3. 対流領域を瞬時混合 (質量加重平均で均一化)
      を ZAMS から age までΔt刻みで進める。
      重力熱項 ε_grav = -T ∂s/∂t は主系列で |ε_grav|/ε≲1e-3 のため無視し、
      静的ソルバーをインナーループにそのまま使う。

  (b) 太陽較正:
      (α_MLT, Y₀, Z₀) を Newton 反復で調整し、age=4.57 Gyr で
        L = L☉,  R = R☉,  (Z/X)_surf = 観測値 (≈0.0245, GN93)
      を満たす標準太陽モデルを作る。

【重要】静的ソルブ 1 回が ~12 分かかるため、完全な 4.57 Gyr 進化 (~50-100
ステップ) はこのサンドボックスでは非現実的。ローカル (16 コア) 実行を想定した
ドライバとして提供する。ここでは
  - 核燃焼率 dX/dt の物理検証 (別途 verify)
  - 少数ステップの自己無撞着発展の実証 (--demo-steps)
  - 較正 Newton の 1 反復
が可能。

使い方:
  # 少数ステップの自己無撞着発展デモ (warm start, 各ステップ低 nfev)
  python3 stellar_evolution_phaseF.py --demo-steps 3 --dt-gyr 0.3

  # 完全発展 (ローカル向け; 時間がかかる)
  python3 stellar_evolution_phaseF.py --evolve --age-gyr 4.57 --n-steps 60

  # 太陽較正 (完全発展を内包; ローカル向け)
  python3 stellar_evolution_phaseF.py --calibrate --newton-steps 4
"""
import os, sys, time, argparse
import importlib.util
import numpy as np

# ── 静的ソルバーのロード ──
# 構造ファイルは環境で名前が異なりうる (番号規約で phase6-10, phase7-0, … と
# リネームされる)。次の優先順で決定する:
#   1. 環境変数 SS_STRUCTURE_FILE
#   2. コマンドライン --structure-file (ここでは簡易に sys.argv を覗く)
#   3. 同ディレクトリの stellar_structure_phase*.py のうち Phase E 機能
#      (set_composition_profile) を含む最新版 (phase 番号最大)
_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_structure_file():
    import glob, re
    # 1. 環境変数
    env = os.environ.get("SS_STRUCTURE_FILE")
    if env and os.path.exists(env):
        return env
    # 2. CLI 引数 (argparse 前に軽く覗く)
    for i, a in enumerate(sys.argv):
        if a == "--structure-file" and i + 1 < len(sys.argv):
            if os.path.exists(sys.argv[i + 1]):
                return sys.argv[i + 1]
        if a.startswith("--structure-file="):
            p = a.split("=", 1)[1]
            if os.path.exists(p):
                return p
    # 3. 自動検出: Phase E 機能を含むものを phase 番号で選ぶ
    cands = glob.glob(os.path.join(_DIR, "stellar_structure_phase*.py"))
    def has_comp(path):
        try:
            with open(path) as f:
                return "def set_composition_profile" in f.read()
        except Exception:
            return False
    def phase_key(path):
        m = re.search(r"phase(\d+)[-_](\d+)", os.path.basename(path))
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    comp_ok = sorted([p for p in cands if has_comp(p)], key=phase_key)
    if comp_ok:
        return comp_ok[-1]                       # 最新の Phase E+ 版
    # フォールバック: 既定名
    return os.path.join(_DIR, "stellar_structure_phase6-10.py")


_SS_PATH = _find_structure_file()
_spec = importlib.util.spec_from_file_location("sscode", _SS_PATH)
ss = importlib.util.module_from_spec(_spec)
sys.modules["sscode"] = ss
_spec.loader.exec_module(ss)
print(f"[Phase F] 静的ソルバー: {os.path.basename(_SS_PATH)}")

# ── バージョン検査: Phase E (非一様組成) 機能が必要 ──
_required = ["set_composition_profile", "load_composition_profile", "_comp_at"]
_missing = [name for name in _required if not hasattr(ss, name)]
if _missing:
    raise ImportError(
        f"読み込んだ {os.path.basename(_SS_PATH)} に Phase E の組成プロファイル\n"
        f"機能が見つかりません (欠落: {', '.join(_missing)})。\n"
        f"  対象ファイル: {_SS_PATH}\n"
        "Phase E で配布した組成対応版 (set_composition_profile / _comp_at を含む)\n"
        "を使ってください。環境変数 SS_STRUCTURE_FILE か --structure-file で\n"
        "明示指定もできます。例:\n"
        "  SS_STRUCTURE_FILE=stellar_structure_phase7-0.py python3 "
        "stellar_evolution_phaseF.py --evolve\n"
        "  python3 stellar_evolution_phaseF.py --structure-file "
        "stellar_structure_phase7-0.py --evolve")

# ── 核物理定数 ──
MeV   = 1.602176634e-6           # erg
Gyr   = 3.15576e16               # s
Q_PP  = 26.20 * MeV              # ニュートリノ損失後 (pp チェーン)
Q_CNO = 25.00 * MeV              # ニュートリノ損失後 (CNO サイクル)
E_PP  = Q_PP  / (4.0 * ss.m_H)   # erg per gram of H (pp)
E_CNO = Q_CNO / (4.0 * ss.m_H)   # erg per gram of H (CNO)
ZX_SUN_SURF = 0.0245             # 観測 (Z/X)_surf (GN93 系)


# ══════════════════════════════════════════════════════
#  静的構造ソルブ (プログラム制御・warm start 対応)
# ══════════════════════════════════════════════════════
def solve_structure(q, X_prof, Z_prof, M_hat, alpha, args,
                    X0_guess=None, priors='none',
                    fit_dm=1e-4, max_nfev=60, two_stage=False, verbose=True):
    """与えられた組成 X(q),Z(q) で静的構造を解く。

    priors: 'zams'   … ZAMS 目標 (L≈0.72, R≈0.87 for M=1) の prior を課す
                        (発展の初期一様モデル用)
            'present'… L=1, R=1 の prior (現在太陽の単発解用)
            'none'   … prior なし。L はエネルギー整合、R は warm start が決める
                        (発展の 2 ステップ目以降; 物理主導)

    Returns dict(sol, T, rho, P, eps_pp, eps_cno, conv_mask, R, L, Tc, rhoc,
                 X0_next)  ← X0_next は次ステップの warm start 用。
    """
    N = len(q)
    # 組成プロファイルを設定
    ss.set_composition_profile(dict(q=q, X=X_prof, Z=Z_prof, XCNO_frac=0.7))
    # グローバル(表面)組成 = プロファイル末端
    Xs = float(X_prof[-1]); Zs = float(Z_prof[-1])
    ss.apply_composition(Xs, max(1.0 - Xs - Zs, 0.0), Zs)
    ss.M_hat = float(M_hat)
    args.mlt_alpha = alpha
    args.opal_X = Xs; args.opal_Z = Zs
    if priors == 'present':
        args.L_zams_solar = 1.0; args.R_zams_solar = 1.0
        args.w_ml_prior = 0.5; args.w_R_prior = 1.5
    elif priors == 'zams':
        args.L_zams_solar = 0.72; args.R_zams_solar = 0.87
        args.w_ml_prior = 1.0; args.w_R_prior = 3.0
    else:  # 'none' — 物理主導 (エネルギー整合 + warm start)
        args.w_ml_prior = 0.0; args.w_R_prior = 0.0
    # 初期値
    if X0_guess is None:
        q_full = ss.make_q_mesh(max(N, 80))
        S0, e0, p0, t0, lLe0 = ss.make_initial_model6_envelope(q_full, M_hat, args)
        X0 = np.concatenate([ss.pack6(
            np.interp(q, q_full, S0), np.interp(q, q_full, e0),
            np.interp(q, q_full, p0), np.interp(q, q_full, t0),
            lLe0), [float(S0[-1])]])
    else:
        X0 = X0_guess.copy()
    lo6, hi6 = ss.make_bounds6(N)
    lo = np.concatenate([lo6, [np.log(0.3)]]); hi = np.concatenate([hi6, [np.log(3.0)]])
    X0 = np.clip(X0, lo, hi)

    from scipy.optimize import least_squares as _lsq
    tab = ss.EnvelopeTable(args, fit_dm=fit_dm, n=4, half_R=0.04, half_L=0.12)
    if not tab._build(float(X0[-1]), float(X0[4 * N])):
        raise RuntimeError("envelope build 失敗")
    _rf = lambda Xv, qq, aa: ss.residual_vector7(Xv, qq, aa, tab)
    t0 = time.time()
    sol = _lsq(_rf, X0, args=(q, args), bounds=(lo, hi), method="trf",
               jac="2-point", tr_solver="lsmr", x_scale="jac",
               ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=max_nfev, verbose=0)
    if two_stage:
        tab2 = ss.EnvelopeTable(args, fit_dm=fit_dm, n=4, half_R=0.012, half_L=0.035)
        if tab2._build(float(sol.x[-1]), float(sol.x[4 * N])):
            _rf2 = lambda Xv, qq, aa: ss.residual_vector7(Xv, qq, aa, tab2)
            sol = _lsq(_rf2, sol.x, args=(q, args), bounds=(lo, hi), method="trf",
                       jac="2-point", tr_solver="lsmr", x_scale="jac",
                       ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=40, verbose=0)
    # 導出量 (ノード上)
    S, ell, p, t, lLe = ss.unpack6(sol.x[:-1], N)
    lR = float(sol.x[-1]); lL = float(lLe)
    Xn, Yn, Zn, mun, XCNOn = ss._comp_at(q)
    P_n = ss.P_REF * np.exp(p); T_n = ss.T_REF * np.exp(t)
    rho_n = ss.density_from_PT(P_n, T_n, mu_loc=mun)
    pp, cno = ss.energy_generation_components(rho_n, T_n, X_loc=Xn, XCNO_loc=XCNOn)
    # 対流判定 (ノード)
    _, nr7, na7 = ss.choose_nabla7(q, S, p, ell, t, lL, args)
    conv = nr7 > na7
    L_star = ss.L_sun * np.exp(lL) * float(ell[-1])
    T_c = ss.T_REF * np.exp(t[0]); P_c = ss.P_REF * np.exp(p[0])
    _Xc, _Yc, _Zc, _muc, _ = ss._comp_at(np.array([q[0]]))
    rho_c = float(ss.density_from_PT(np.array([P_c]), np.array([T_c]), mu_loc=_muc)[0])
    env = ss.integrate_envelope(np.exp(lR), np.exp(lL), args, fit_dm=fit_dm)
    r_node = np.exp(S)                      # r/R☉ (ノード)
    R_star = float(np.exp(lR))
    # 対流境界: 表面対流層底 (r_cz/R*) と 対流コア頂点 (q_core)
    r_cz = np.nan; q_cc = np.nan
    if np.any(conv):
        # 表面から内側へ連続する対流ブロックの底
        if conv[-1]:
            k = len(conv) - 1
            while k > 0 and conv[k - 1]:
                k -= 1
            r_cz = r_node[k] / R_star
        # 中心から外へ連続する対流ブロックの頂点 (対流コア)
        if conv[0]:
            j = 0
            while j < len(conv) - 1 and conv[j + 1]:
                j += 1
            q_cc = float(q[j])
    res_inf = float(np.max(np.abs(sol.fun)))
    if verbose:
        print(f"    solve: {time.time()-t0:.0f}s nfev={sol.nfev} "
              f"|res|∞={res_inf:.2e}  "
              f"R={R_star:.4f} L={L_star/ss.L_sun:.4f} Tc={T_c/1e6:.2f}MK")
    return dict(sol=sol, X0_next=sol.x, T=T_n, rho=rho_n, P=P_n, r=r_node,
                eps_pp=pp, eps_cno=cno, conv_mask=conv,
                R=R_star, L=L_star / ss.L_sun,
                Tc=T_c, rhoc=rho_c,
                Teff=(env['T_eff'] if env else np.nan),
                Xsurf=Xs, Zsurf=Zs, q=q,
                r_cz=r_cz, q_core=q_cc,
                res_inf=res_inf, nfev=int(sol.nfev),
                Lmass_ratio=_energy_balance_ratio(q, T_n, rho_n, Xn, XCNOn,
                                                  L_star, M_hat))


def _energy_balance_ratio(q, T, rho, X, XCNO, L_star, M_hat):
    """∫εdm / L_star を返す (熱平衡のチェック)。"""
    pp, cno = ss.energy_generation_components(rho, T, X_loc=X, XCNO_loc=XCNO)
    eps = pp + cno
    Lm = float(np.sum(0.5 * (eps[:-1] + eps[1:]) * np.diff(q))) * M_hat * ss.M_sun
    return Lm / max(L_star, 1e-30)


# ══════════════════════════════════════════════════════
#  核燃焼 + 混合 による組成更新
# ══════════════════════════════════════════════════════
def nuclear_update(q, X, Z, eps_pp, eps_cno, conv_mask, dt):
    """dX/dt = -(ε_pp/E_pp + ε_CNO/E_CNO) を陽的に 1 ステップ進め、
    対流領域を質量加重平均で瞬時混合する。Z は不変 (金属は燃えない)。"""
    dXdt = -(eps_pp / E_PP + eps_cno / E_CNO)
    Xnew = np.clip(X + dXdt * dt, 0.0, X)            # H は減る一方
    # 対流領域の混合 (質量重み dq)
    dq = np.gradient(q)
    if conv_mask is not None and np.any(conv_mask):
        # 連続する対流ブロックごとに均一化
        idx = np.where(conv_mask)[0]
        # ブロック分割
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for blk in splits:
            w = np.abs(dq[blk])
            if w.sum() > 0:
                Xnew[blk] = np.sum(Xnew[blk] * w) / w.sum()
    return Xnew


# ══════════════════════════════════════════════════════
#  時間発展ループ
# ══════════════════════════════════════════════════════
def evolve(M_hat=1.0, X0=0.70, Z0=0.02, alpha=1.8, age_gyr=4.57,
           n_steps=60, N_mesh=97, max_nfev=60, demo_steps=None,
           X0_guess=None, verbose=True, save_profiles=False):
    """ZAMS (一様 X0,Z0) から age_gyr まで主系列進化。
    demo_steps を指定するとそのステップ数だけ進めて途中経過を返す。
    save_profiles=True で各ステップの X(q),T(q),ρ(q) プロファイルを記録。"""
    q = ss.make_q_mesh_fit(N_mesh, fit_dm=1e-4)
    args = ss.build_default_args()
    args.mlt_alpha = alpha
    X = np.full(len(q), float(X0))
    Z = np.full(len(q), float(Z0))
    dt = (age_gyr / n_steps) * Gyr
    total = demo_steps if demo_steps is not None else n_steps
    age = 0.0
    guess = X0_guess
    hist = []
    profiles = [] if save_profiles else None
    for step in range(total):
        if verbose:
            print(f"  [step {step+1}/{total}] age={age/Gyr:.3f} Gyr "
                  f"X_c={X[0]:.4f}")
        # step 0 は ZAMS 一様モデル → ZAMS prior。以降は warm start + 物理主導。
        if step == 0:
            pr = 'zams'
            nf = max_nfev if guess is not None else 400
        else:
            pr = 'none'; nf = max_nfev
        res = solve_structure(q, X, Z, M_hat, alpha, args,
                              X0_guess=guess, priors=pr, max_nfev=nf,
                              verbose=verbose)
        guess = res["X0_next"]
        hist.append(dict(step=step + 1, age=age / Gyr, Xc=float(X[0]),
                         Xsurf=res["Xsurf"], Zsurf=res["Zsurf"],
                         R=res["R"], L=res["L"], Tc=res["Tc"] / 1e6,
                         rhoc=res["rhoc"], Teff=res["Teff"],
                         r_cz=res["r_cz"], q_core=res["q_core"],
                         Lmass_ratio=res["Lmass_ratio"],
                         res_inf=res["res_inf"], nfev=res["nfev"]))
        if save_profiles:
            profiles.append(dict(step=step + 1, age=age / Gyr,
                                 X=X.copy(), T=res["T"].copy(),
                                 rho=res["rho"].copy(), r=res["r"].copy(),
                                 conv=res["conv_mask"].copy()))
        # 組成更新
        X = nuclear_update(q, X, Z, res["eps_pp"], res["eps_cno"],
                           res["conv_mask"], dt)
        age += dt
    return dict(q=q, X=X, Z=Z, age=age / Gyr, hist=hist,
                last=res, guess=guess, profiles=profiles)


def write_evolution_track(hist, outfile, meta=None):
    """発展トラックをテキストテーブルに書き出す (1 ステップ 1 行)。"""
    lines = []
    lines.append("# Phase F 主系列進化トラック")
    if meta:
        for k, v in meta.items():
            lines.append(f"# {k}: {v}")
    lines.append("#")
    hdr = ("# {:>4} {:>9} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>9} "
           "{:>8} {:>8} {:>9} {:>6} {:>10}").format(
        "step", "age[Gyr]", "X_c", "X_surf", "Z_surf", "R/Rsun", "L/Lsun",
        "Teff[K]", "Tc[MK]", "rhoc", "r_cz/R*", "q_core", "nfev", "|res|inf")
    lines.append(hdr)
    lines.append("# " + "-" * (len(hdr) - 2))
    for h in hist:
        rcz = f"{h['r_cz']:.4f}" if np.isfinite(h['r_cz']) else "  --  "
        qcc = f"{h['q_core']:.4f}" if np.isfinite(h['q_core']) else "  --  "
        lines.append(
            "  {:>4d} {:>9.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>9.4f} {:>9.4f} "
            "{:>8.0f} {:>9.3f} {:>8.2f} {:>8} {:>9} {:>6d} {:>10.2e}".format(
                h['step'], h['age'], h['Xc'], h['Xsurf'], h['Zsurf'],
                h['R'], h['L'], h['Teff'], h['Tc'], h['rhoc'],
                rcz, qcc, h['nfev'], h['res_inf']))
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════
#  太陽較正 (Newton on α_MLT, Y0, Z0)
# ══════════════════════════════════════════════════════
def solar_calibration(alpha0=1.8, Y0=0.273, age_gyr=4.57, n_steps=60,
                      newton_steps=4, N_mesh=97, zx_target=ZX_SUN_SURF,
                      recompute_jac=False, verbose=True):
    """(α_MLT, Y0) を Newton 反復で調整し、age=age_gyr で L=1, R=1 を満たす。

    (Z/X)_surf は拡散なしでは Z0/X0 のまま発展中不変なので、
    Z0 = zx_target·X0 と代数的に固定する (X0=(1-Y0)/(1+zx_target))。
    → 3 パラメータ問題が (α, Y0) の 2×2 に縮小。3 つ目の目標は構成上自動満足。

    効率化:
      - 各完全発展の ZAMS を前回解から warm start (215s→~40s)。
      - ヤコビアンは既定で最初に 1 度だけ数値計算し以降再利用 (弦 Newton)。
        recompute_jac=True で毎反復再計算 (通常の Newton, より頑健だが高コスト)。

    Returns (params dict, 最終 evolve 出力, 履歴 list)。
    """
    def comp_from_Y(Yv):
        X0 = (1.0 - Yv) / (1.0 + zx_target)
        Z0 = zx_target * X0
        return X0, Z0

    _zams_guess = [None]   # ZAMS warm start キャッシュ (可変クロージャ)

    def objective(a, Yv):
        X0, Z0 = comp_from_Y(Yv)
        out = evolve(1.0, X0, Z0, a, age_gyr, n_steps, N_mesh,
                     X0_guess=_zams_guess[0], verbose=False)
        # 最初の評価で得た ZAMS 相当解を以降の warm start に (step1 の解)
        if _zams_guess[0] is None and out.get("guess") is not None:
            # out['guess'] は最終ステップの解。ZAMS 用には step1 の解が欲しいが、
            # 近傍性から最終解でも十分な初期値になる (composition のみ差)。
            _zams_guess[0] = out["guess"]
        r = out["last"]
        F = np.array([np.log(max(r["L"], 1e-9)), np.log(max(r["R"], 1e-9))])
        return F, out

    p = np.array([alpha0, Y0])
    hist = []
    J = None
    dp = np.array([0.10, 0.010])   # 差分幅 (α, Y0)
    for it in range(newton_steps):
        F, out = objective(p[0], p[1])
        r = out["last"]
        X0, Z0 = comp_from_Y(p[1])
        rec = dict(it=it, alpha=p[0], Y0=p[1], X0=X0, Z0=Z0,
                   L=r["L"], R=r["R"], Tc=r["Tc"] / 1e6, Teff=r["Teff"],
                   Xc=float(out["X"][0]), Fnorm=float(np.linalg.norm(F)))
        hist.append(rec)
        if verbose:
            print(f"[calib {it}] α={p[0]:.4f} Y0={p[1]:.4f} "
                  f"(X0={X0:.4f} Z0={Z0:.4f}) → L={r['L']:.4f} R={r['R']:.4f} "
                  f"Tc={r['Tc']/1e6:.2f}MK  |F|={np.linalg.norm(F):.3e}")
        if np.linalg.norm(F) < 2e-3:
            print("[calib] 収束"); break
        # ヤコビアン (初回のみ, または recompute_jac)
        if J is None or recompute_jac:
            J = np.zeros((2, 2))
            for k in range(2):
                pk = p.copy(); pk[k] += dp[k]
                Fk, _ = objective(pk[0], pk[1])
                J[:, k] = (Fk - F) / dp[k]
            if verbose:
                print(f"    Jacobian:\n      ∂lnL/∂α={J[0,0]:+.3f} ∂lnL/∂Y0={J[0,1]:+.3f}\n"
                      f"      ∂lnR/∂α={J[1,0]:+.3f} ∂lnR/∂Y0={J[1,1]:+.3f}")
        try:
            step = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, F, rcond=None)[0]
        p = p - step
        p[0] = np.clip(p[0], 1.0, 2.6)     # α
        p[1] = np.clip(p[1], 0.22, 0.32)   # Y0
    X0f, Z0f = comp_from_Y(p[1])
    params = dict(alpha_MLT=float(p[0]), Y0=float(p[1]), X0=float(X0f),
                  Z0=float(Z0f), zx_surf=zx_target)
    return params, out, hist


def main():
    pa = argparse.ArgumentParser(description="Phase F: 主系列進化 + 太陽較正")
    pa.add_argument("--demo-steps", type=int, default=None,
                    help="少数ステップの自己無撞着発展デモ")
    pa.add_argument("--dt-gyr", type=float, default=0.3, help="デモの Δt [Gyr]")
    pa.add_argument("--evolve", action="store_true", help="完全発展")
    pa.add_argument("--calibrate", action="store_true", help="太陽較正")
    pa.add_argument("--age-gyr", type=float, default=4.57)
    pa.add_argument("--n-steps", type=int, default=60)
    pa.add_argument("--newton-steps", type=int, default=4)
    pa.add_argument("--recompute-jac", action="store_true",
                    help="較正で毎反復ヤコビアンを再計算 (既定は初回のみ=弦Newton, 高速)")
    pa.add_argument("--M", type=float, default=1.0)
    pa.add_argument("--X0", type=float, default=0.70)
    pa.add_argument("--Z0", type=float, default=0.02)
    pa.add_argument("--alpha", type=float, default=1.8)
    pa.add_argument("--n-mesh", type=int, default=97)
    pa.add_argument("--structure-file", type=str, default=None,
                    help="静的ソルバーのファイル名を明示指定 "
                         "(既定: 組成対応版を自動検出)")
    pa.add_argument("--save-track", type=str, default=None,
                    help="発展トラック (1 ステップ 1 行の要約テーブル) の保存先。"
                         " 既定: --evolve 時は phaseF_track.txt に自動保存")
    pa.add_argument("--save-profiles", type=str, default=None,
                    help="各ステップの構造プロファイル X(q),T,ρ,r,対流 を "
                         "npz に保存 (例: phaseF_profiles.npz)")
    pa.add_argument("--out-prefix", type=str, default="phaseF",
                    help="出力ファイル接頭辞 (既定: phaseF)")
    cli = pa.parse_args()

    if cli.calibrate:
        # --calibrate は --evolve より優先 (両方指定時も較正を実行)
        from datetime import datetime as _dt
        _t0 = _dt.now()
        print(f"[Phase F] 太陽較正 開始: {_t0:%Y-%m-%d %H:%M:%S}")
        Y0_init = 1.0 - cli.X0 - cli.Z0
        print(f"[Phase F] 初期: α={cli.alpha} Y0={Y0_init:.4f}  "
              f"目標 L=1, R=1, (Z/X)_surf={ZX_SUN_SURF} "
              f"(age={cli.age_gyr}Gyr, {cli.n_steps}steps, "
              f"Jac={'毎回' if cli.recompute_jac else '初回のみ(弦Newton)'})")
        params, out, chist = solar_calibration(
            cli.alpha, Y0_init, cli.age_gyr, cli.n_steps, cli.newton_steps,
            cli.n_mesh, recompute_jac=cli.recompute_jac)
        _t1 = _dt.now(); _dur = (_t1 - _t0).total_seconds()
        r = out["last"]
        print("═" * 64)
        print(f"[Phase F] 較正結果:")
        print(f"  α_MLT = {params['alpha_MLT']:.4f}")
        print(f"  Y0    = {params['Y0']:.4f}   X0 = {params['X0']:.4f}   "
              f"Z0 = {params['Z0']:.4f}   (Z/X)_surf = {params['zx_surf']:.4f}")
        print(f"  → age={out['age']:.2f}Gyr: L={r['L']:.4f} R={r['R']:.4f} "
              f"Tc={r['Tc']/1e6:.3f}MK Teff={r['Teff']:.0f}K X_c={out['X'][0]:.4f}")
        print(f"[Phase F] 較正終了: {_t1:%Y-%m-%d %H:%M:%S}  "
              f"所要 {_dur:.1f}秒 ({_dur/60:.1f}分)")
        # 較正履歴 + 最終トラック
        with open(os.path.join(_DIR, f"{cli.out_prefix}_calib.txt"), "w") as f:
            f.write("# Phase F 太陽較正履歴\n")
            f.write(f"# 開始 {_t0:%Y-%m-%d %H:%M:%S} 終了 {_t1:%Y-%m-%d %H:%M:%S} "
                    f"所要 {_dur:.1f}秒\n")
            f.write("# it   alpha      Y0      X0      Z0     L       R      "
                    "Tc[MK]  Teff   X_c     |F|\n")
            for h in chist:
                f.write(f"  {h['it']:>2d} {h['alpha']:.4f} {h['Y0']:.4f} "
                        f"{h['X0']:.4f} {h['Z0']:.4f} {h['L']:.4f} {h['R']:.4f} "
                        f"{h['Tc']:.3f} {h['Teff']:.0f} {h['Xc']:.4f} "
                        f"{h['Fnorm']:.2e}\n")
        write_evolution_track(out["hist"],
                              os.path.join(_DIR, f"{cli.out_prefix}_calib_track.txt"),
                              meta={"alpha_MLT": params['alpha_MLT'],
                                    "Y0": params['Y0'], "X0": params['X0'],
                                    "Z0": params['Z0'], "type": "calibrated"})
        np.savez(os.path.join(_DIR, f"{cli.out_prefix}_calib_evolved.npz"),
                 q=out["q"], X=out["X"], Z=out["Z"], age=out["age"],
                 **params)
        print(f"[Phase F] 較正履歴/トラック/最終X(q) を {cli.out_prefix}_calib* に保存")
    elif cli.demo_steps is not None:
        n_eff = max(1, int(round(cli.age_gyr / cli.dt_gyr)))
        out = evolve(cli.M, cli.X0, cli.Z0, cli.alpha, cli.age_gyr,
                     n_steps=n_eff, demo_steps=cli.demo_steps,
                     N_mesh=cli.n_mesh)
        print("\n=== 発展履歴 ===")
        for h in out["hist"]:
            print(f"  age={h['age']:.3f} Gyr  X_c={h['Xc']:.4f}  "
                  f"R={h['R']:.4f}  L={h['L']:.4f}  Tc={h['Tc']:.2f}MK  "
                  f"Teff={h['Teff']:.0f}K")
        np.savez(os.path.join(_DIR, "phaseF_demo.npz"),
                 q=out["q"], X=out["X"], Z=out["Z"], age=out["age"])
        print(f"\n最終 X(q) を phaseF_demo.npz に保存 (age={out['age']:.2f} Gyr)")
    elif cli.evolve:
        from datetime import datetime as _dt
        _t0 = _dt.now()
        print(f"[Phase F] 計算開始: {_t0:%Y-%m-%d %H:%M:%S}")
        print(f"[Phase F] M={cli.M} X0={cli.X0} Z0={cli.Z0} α={cli.alpha} "
              f"age={cli.age_gyr} Gyr / {cli.n_steps} steps (N={cli.n_mesh})")
        want_prof = cli.save_profiles is not None
        out = evolve(cli.M, cli.X0, cli.Z0, cli.alpha, cli.age_gyr,
                     cli.n_steps, cli.n_mesh, save_profiles=want_prof)
        _t1 = _dt.now(); _dur = (_t1 - _t0).total_seconds()
        r = out["last"]
        # 最終状態のサマリ
        print("═" * 64)
        print(f"完全発展完了 age={out['age']:.3f} Gyr")
        print(f"  最終: X_c={out['X'][0]:.4f}  X_surf={r['Xsurf']:.4f}  "
              f"R={r['R']:.4f} R☉  L={r['L']:.4f} L☉")
        print(f"        T_eff={r['Teff']:.0f} K  T_c={r['Tc']/1e6:.3f} MK  "
              f"ρ_c={r['rhoc']:.1f}  ∫εdm/L={r['Lmass_ratio']:.4f}")
        rcz = r['r_cz']
        if np.isfinite(rcz):
            print(f"        表面対流層底 r_cz/R*={rcz:.4f}")
        print(f"[Phase F] 計算終了: {_t1:%Y-%m-%d %H:%M:%S}")
        print(f"[Phase F] 所要時間: {_dur:.1f} 秒 ({_dur/60:.1f} 分, "
              f"{_dur/max(cli.n_steps,1):.1f} 秒/ステップ)")
        # 最終 X(q)
        np.savez(os.path.join(_DIR, f"{cli.out_prefix}_evolved.npz"),
                 q=out["q"], X=out["X"], Z=out["Z"], age=out["age"])
        # 発展トラック (要約テーブル)
        track = cli.save_track or os.path.join(_DIR, f"{cli.out_prefix}_track.txt")
        write_evolution_track(out["hist"], track, meta={
            "M": cli.M, "X0": cli.X0, "Z0": cli.Z0, "alpha_MLT": cli.alpha,
            "age_gyr": cli.age_gyr, "n_steps": cli.n_steps, "N_mesh": cli.n_mesh,
            "開始": f"{_t0:%Y-%m-%d %H:%M:%S}", "終了": f"{_t1:%Y-%m-%d %H:%M:%S}",
            "所要秒": f"{_dur:.1f}"})
        print(f"[Phase F] 発展トラックを保存: {track}")
        # 構造プロファイル (各ステップ)
        if want_prof and out["profiles"]:
            prof = out["profiles"]
            np.savez(os.path.join(_DIR, cli.save_profiles),
                     steps=np.array([p["step"] for p in prof]),
                     ages=np.array([p["age"] for p in prof]),
                     q=out["q"],
                     X=np.array([p["X"] for p in prof]),
                     T=np.array([p["T"] for p in prof]),
                     rho=np.array([p["rho"] for p in prof]),
                     r=np.array([p["r"] for p in prof]),
                     conv=np.array([p["conv"] for p in prof]))
            print(f"[Phase F] 構造プロファイル ({len(prof)} ステップ) を保存: "
                  f"{cli.save_profiles}")
    else:
        pa.print_help()


if __name__ == "__main__":
    main()
