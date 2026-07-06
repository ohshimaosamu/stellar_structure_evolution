#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_phase_e.py — Phase E 検証プロトコル
  ベンチマーク (i)   太陽 ZAMS の対流層底
  ベンチマーク (ii)  M=0.3 M☉ が (ほぼ) 完全対流
  ベンチマーク (iii) M=1.5 M☉ で対流コア q_core≈0.06–0.08 + 表面対流消失
  ベンチマーク (iv)  【新規】文献 X(q)(Model S)を入れた静的現在太陽モデルの較正
                     → L≈1 L☉, T_c≈15.6 MK, r_cz≈0.71–0.73 R☉
  検証 (v)           prior (w_ml, w_R) を段階的にゼロへ落として素の物理解を確認

各ベンチマークはソルバー実行を伴い時間がかかるため、このスクリプトは
既存の出力テーブル (ph10*_table.txt) を解析して判定する形にしている。
実行方法は末尾の docstring を参照。
"""
import sys, os, glob
import numpy as np


def read_table(path):
    """テーブル出力 (# メタ + 数値列 + 末尾 r/c フラグ) を読む。"""
    meta = {}
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('#'):
                for key, tag in [('R=', 'R'), ('L=', 'L'), ('T_c=', 'Tc'),
                                 ('M=', 'M')]:
                    pass
                meta.setdefault('_hdr', []).append(s)
                continue
            if not s:
                continue
            parts = s.split()
            try:
                vals = [float(x) for x in parts[:-1]]
                flag = parts[-1]
            except ValueError:
                try:
                    vals = [float(x) for x in parts]
                    flag = ''
                except ValueError:
                    continue
            rows.append((vals, flag))
    return meta, rows


def parse_header(meta):
    out = {}
    for line in meta.get('_hdr', []):
        for token in line.replace('#', ' ').split():
            for k in ['M=', 'R=', 'L=', 'T_c=', 'rho_c=']:
                if token.startswith(k):
                    try:
                        out[k.rstrip('=')] = float(token[len(k):]
                                                   .replace('R_sun','')
                                                   .replace('L_sun','')
                                                   .replace('MK',''))
                    except ValueError:
                        pass
    return out


def analyze(path, label):
    meta, rows = read_table(path)
    rows = [(v, f) for v, f in rows if len(v) >= 6]
    hdr = parse_header(meta)
    # 列: r/R_sun  M/M_sun  T[K]  P  rho  L/L_sun  flag
    r_Rsun = np.array([v[0] for v, f in rows])
    M_Msun = np.array([v[1] for v, f in rows])
    T  = np.array([v[2] for v, f in rows])
    flags = [f for v, f in rows]
    conv = np.array([f == 'c' for f in flags])
    Rstar = hdr.get('R', 1.0)          # R*/R_sun
    rR = r_Rsun / max(Rstar, 1e-9)     # r/R*
    Mstar = M_Msun.max() if M_Msun.size else 1.0
    q = M_Msun / max(Mstar, 1e-9)      # q = m/M*
    print(f"\n[{label}] {os.path.basename(path)}")
    for k, val in hdr.items():
        print(f"   {k} = {val}")
    if conv.any():
        idx = np.argsort(rR)
        rRs = rR[idx]; qs = q[idx]; Ts = T[idx]; cs = conv[idx]
        k = len(cs) - 1
        if cs[k]:
            while k > 0 and cs[k-1]:
                k -= 1
            print(f"   表面対流層底: r/R*={rRs[k]:.4f}  q={qs[k]:.5f}  T={Ts[k]:.3e} K")
        if cs[0]:
            j = 0
            while j < len(cs)-1 and cs[j+1]:
                j += 1
            print(f"   対流コア頂点: q_core={qs[j]:.4f}  r/R*={rRs[j]:.4f}")
        print(f"   対流点の割合: {conv.sum()/len(conv)*100:.1f}% (メッシュ点)")
    else:
        print("   対流層なし (全点放射)")
    return hdr, rR, q, T, conv


def main():
    files = sorted(glob.glob('ph10*_table.txt'))
    if not files:
        print("テーブルファイル (ph10*_table.txt) が見つかりません。")
        return
    print("=" * 70)
    print("Phase E ベンチマーク解析")
    print("=" * 70)
    for f in files:
        lbl = 'present-Sun' if 'sun' in f.lower() else 'ZAMS'
        analyze(f, lbl)

    # ベンチマーク (iv) の合否判定 (present-Sun テーブルがある場合)
    sun = [f for f in files if 'sun' in f.lower()]
    if sun:
        hdr, rR, q, T, conv = analyze(sun[0], 'present-Sun 判定')
        print("\n" + "=" * 70)
        print("ベンチマーク (iv) 現在太陽モデルの合否")
        print("=" * 70)
        L = hdr.get('L'); Tc = hdr.get('T_c'); R = hdr.get('R')
        # r_cz
        idx = np.argsort(rR)
        rRs = rR[idx]; cs = conv[idx]
        r_cz = None
        k = len(cs) - 1
        if cs[k]:
            while k > 0 and cs[k-1]:
                k -= 1
            r_cz = rRs[k]
        def ok(name, val, lo, hi, unit=''):
            s = 'OK' if (val is not None and lo <= val <= hi) else '要確認'
            vs = f"{val:.4f}" if val is not None else 'N/A'
            print(f"   {name:20s}: {vs}{unit}  (目標 {lo}–{hi}{unit})  [{s}]")
        ok('L / L☉', L, 0.95, 1.05)
        ok('R / R☉', R, 0.98, 1.02)
        ok('T_c (MK)', Tc, 15.0, 16.2, ' MK')
        ok('r_cz / R*', r_cz, 0.70, 0.74)


if __name__ == '__main__':
    main()
