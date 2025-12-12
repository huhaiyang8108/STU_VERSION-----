#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STU 潮汐反演（S 极窄范围锁定版）
----------------------------------------------------
修改点：
1. S_GLOBAL_RANGE_LOG = 0.02：强制 S 全程波动不超过 ±5%。
2. cfg["S"] = (4e-5, 6e-5)：硬性物理范围进一步缩窄。
3. TRUST_RADIUS = 0.05：增加逐日惯性。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from scipy.special import kv
from scipy.optimize import minimize
import optuna
import os
import tkinter as tk
from tkinter import filedialog
import warnings

warnings.filterwarnings("ignore")

# 全局绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 用户参数 (核心调优区) =================

HALF_WINDOW = 3   
COARSE_TRIALS = 30
COARSE_POINTS = 20

# 1. 信任域 (逐日变化限制)
# 改为 0.05 (允许每天变化 12%)，增加稳定性
TRUST_RADIUS = 0.05

# 2. S 的全局波动范围限制 (Log10单位)
# 改为 0.02 (允许全程波动 ±4.7%)，极度稳定，接近直线
S_GLOBAL_RANGE_LOG = 0.02  

# ================= 物理模型 =================
rc=0.171/2; rw=0.150/2; bp=26.26; b=40.44; S_S=1e-5; zi=-29.8; wm2=2*np.pi/0.5175/86400

# 粗反演范围
COARSE_BOUNDS = dict(S=(1e-5, 9e-5), T=(1e-6, 1e-2), U=(1e-10, 1e-5))

def select_data_file():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="选择数据文件")
    if not file_path: raise FileNotFoundError("未选择文件")
    return file_path

def forward(x, AR, pin):
    S, T, U = x
    if S <= 0 or T <= 0 or U <= 0: return np.array([1e6, 1e6, 1e6])
    Ss, RKuB = S/b, (S*bp)/(b*S_S); K_K, D_D = U*bp, U*(bp**2)/S_S
    delta = np.sqrt(2*D_D/wm2); A = 1j*wm2*S/T; B = K_K/T
    C = (1+1j)/delta; D_val = 1/np.tanh(C*zi); belta = np.sqrt(A - B*C*D_val)
    E = 1j*wm2*rc**2 * kv(0, belta*rw); F = 2*T*belta*rw * kv(1, belta*rw)
    sigma = 1 + E/F; N = 1 - np.cosh(C*zi); L = np.sinh(C*zi)
    H = S*1j*wm2 + K_K*C*RKuB*(N/L); M = sigma*(S*1j*wm2 - K_K*C*(1/np.tanh(C*zi)))
    hw = H/M; amp = np.abs(hw); pha = np.angle(hw)*180/np.pi
    q1 = amp - AR*Ss; q2 = pha - pin
    return np.array([q1, q2, q1+q2])

def misfit_window_gaussian(x, AR_chunk, pin_chunk):
    S, T, U = x
    if S <= 0 or T <= 0 or U <= 0: return 1e99
    n = len(AR_chunk)
    sigma = n / 4.0  
    center = (n - 1) / 2.0
    indices = np.arange(n)
    weights = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
    err = 0.0; weight_sum = 0.0
    for i in range(n):
        q1, q2, _ = forward(x, AR_chunk[i], pin_chunk[i])
        w = weights[i]
        err += (q1**2 + q2**2) * w
        weight_sum += w
    return err / weight_sum

def calculate_global_median_S(AR, pin, cfg):
    print(">>> 正在计算全局 S 中值...")
    step = max(1, len(AR) // 30)
    S_list = []
    def obj(trial):
        S = trial.suggest_float("S", cfg["S"][0], cfg["S"][1], log=True)
        T = trial.suggest_float("T", cfg["T"][0], cfg["T"][1], log=True)
        U = trial.suggest_float("U", cfg["U"][0], cfg["U"][1], log=True)
        q1, q2, _ = forward([S,T,U], ar_val, pin_val)
        return q1**2 + q2**2
    for i in range(0, len(AR), step):
        ar_val, pin_val = AR[i], pin[i]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(obj, n_trials=20, show_progress_bar=False)
        S_list.append(study.best_params["S"])
    S_median = np.median(S_list)
    print(f"    全局 S 中值 = {S_median:.2e}")
    return S_median

def optimize_step_scipy(AR_chunk, pin_chunk, x0, cfg, s_median_log):
    def obj_log(log_x): return misfit_window_gaussian(10**log_x, AR_chunk, pin_chunk)
    log_x0 = np.log10(x0)
    bounds = []
    for k, lx in zip(["S","T","U"], log_x0):
        gmin_phys, gmax_phys = np.log10(cfg[k])
        
        if k == "S":
            # 核心：极窄的动态范围
            gmin_anchor = s_median_log - S_GLOBAL_RANGE_LOG
            gmax_anchor = s_median_log + S_GLOBAL_RANGE_LOG
            gmin = max(gmin_phys, gmin_anchor)
            gmax = min(gmax_phys, gmax_anchor)
        else:
            gmin, gmax = gmin_phys, gmax_phys
            
        bmin = max(gmin, lx - TRUST_RADIUS)
        bmax = min(gmax, lx + TRUST_RADIUS)
        if bmax < bmin: 
            center = (gmin + gmax) / 2
            bmin = max(gmin, center - TRUST_RADIUS)
            bmax = min(gmax, center + TRUST_RADIUS)
        bounds.append((bmin, bmax))
    res = minimize(obj_log, log_x0, method='L-BFGS-B', bounds=bounds)
    return 10**res.x

def optimize_init_optuna(AR_chunk, pin_chunk, cfg):
    def objective(trial):
        S = trial.suggest_float("S", cfg["S"][0], cfg["S"][1], log=True)
        T = trial.suggest_float("T", cfg["T"][0], cfg["T"][1], log=True)
        U = trial.suggest_float("U", cfg["U"][0], cfg["U"][1], log=True)
        return misfit_window_gaussian([S, T, U], AR_chunk, pin_chunk)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    bp = study.best_params
    return np.array([bp["S"], bp["T"], bp["U"]])

def run_directional_pass(AR, pin, cfg, s_median, init_guess=None):
    n = len(AR)
    S_res = np.zeros(n); T_res = np.zeros(n); U_res = np.zeros(n)
    prev_x = init_guess
    s_median_log = np.log10(s_median)
    for i in range(n):
        idx_start = max(0, i - HALF_WINDOW)
        idx_end   = min(n, i + HALF_WINDOW + 1)
        AR_c  = AR[idx_start:idx_end]; pin_c = pin[idx_start:idx_end]
        if prev_x is None: best_x = optimize_init_optuna(AR_c, pin_c, cfg)
        else: best_x = optimize_step_scipy(AR_c, pin_c, prev_x, cfg, s_median_log)
        S_res[i], T_res[i], U_res[i] = best_x
        prev_x = best_x
    return S_res, T_res, U_res

def plot_dense_ticks(dates, S, T, U, Misfit, filename):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.12) 
    data_list = [S, T, U]
    labels = [r'$S$ (Storage)', r'$T$ (Transmissivity)', r'$U$ (Leakage)']
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    for i in range(3):
        ax = axes[i]
        ax.plot(dates, data_list[i], color=colors[i], linewidth=1.5, label=labels[i])
        ax.set_ylabel(labels[i].split()[0], fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        
        # 密集刻度设置
        dense_subs = np.arange(1.0, 10.0, 0.2)
        maj_loc = ticker.LogLocator(base=10.0, subs=dense_subs, numticks=20)
        ax.yaxis.set_major_locator(maj_loc)
        maj_fmt = ticker.FormatStrFormatter('%.1e')
        ax.yaxis.set_major_formatter(maj_fmt)
        
        ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(loc='upper right')

    ax4 = axes[3]
    ax4.plot(dates, Misfit, 'k-', linewidth=1.0, alpha=0.8, label='Misfit')
    ax4.set_ylabel('Misfit', fontsize=12, fontweight='bold')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax4.yaxis.set_major_formatter(formatter)
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax4.grid(True, which='major', linestyle='-', alpha=0.6)
    ax4.tick_params(which='both', direction='in', top=True, right=True)
    ax4.legend(loc='upper right')

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%Y-%m')
    ax4.xaxis.set_major_locator(locator)
    ax4.xaxis.set_major_formatter(formatter)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Inversion Results (S Tightly Locked)', fontsize=14, y=0.92)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已绘图: {filename}")
    plt.show()

def main():
    data_file = select_data_file()
    base = os.path.splitext(os.path.basename(data_file))[0]
    
    dates, HA, HP, EA, EP = [], [], [], [], []
    with open(data_file, "r") as f:
        for line in f:
            p = line.split()
            if len(p)<5: continue
            dates.append(datetime.strptime(p[0], "%Y/%m/%d"))
            HA.append(float(p[1])); HP.append(float(p[2]))
            EA.append(float(p[3])); EP.append(float(p[4]))
    dates = np.array(dates); n = len(dates)
    AR = (np.array(HA)/np.array(EA)) * 1e9
    pin = np.array(HP) - np.array(EP)

    # --------------------------------------------------------
    # 【核心修改区】更严格的参数范围
    # --------------------------------------------------------
    cfg = {
        # 1. 物理硬边界：进一步缩窄到 1e-5 ~ 3e-5
        "S": (1.0e-5, 3.0e-5),  
        
        # T 和 U 保持宽范围
        "T": (1e-7, 1e-2),
        "U": (1e-10, 1e-5)
    }
    
    print(f"\n>>> 强制配置参数范围 (更严格):")
    print(f"    S: {cfg['S']}")
    print(f"    T: {cfg['T']}")
    print(f"    U: {cfg['U']}")

    # 1. 计算中值
    s_median = calculate_global_median_S(AR, pin, cfg)

    # 2. Forward
    print("\n>>> 1/3 Forward Pass...")
    Sf, Tf, Uf = run_directional_pass(AR, pin, cfg, s_median, init_guess=None)
    
    # 3. Backward
    print(">>> 2/3 Backward Pass...")
    AR_rev, pin_rev = AR[::-1], pin[::-1]
    init_rev = [Sf[-1], Tf[-1], Uf[-1]]
    Sb_r, Tb_r, Ub_r = run_directional_pass(AR_rev, pin_rev, cfg, s_median, init_guess=init_rev)
    Sb, Tb, Ub = Sb_r[::-1], Tb_r[::-1], Ub_r[::-1]

    # 4. Fusion
    print(">>> 3/3 Fusion...")
    w_fwd = np.linspace(0, 1, n)
    w_bwd = 1.0 - w_fwd
    S_fin = 10**(np.log10(Sf)*w_fwd + np.log10(Sb)*w_bwd)
    T_fin = 10**(np.log10(Tf)*w_fwd + np.log10(Tb)*w_bwd)
    U_fin = 10**(np.log10(Uf)*w_fwd + np.log10(Ub)*w_bwd)

    q_res = np.zeros((n, 3))
    for i in range(n):
        q1, q2, q3 = forward([S_fin[i], T_fin[i], U_fin[i]], AR[i], pin[i])
        q_res[i] = [q1, q2, q3]

    OUT_DATA = f"STU_StrictS_{base}.dat"
    np.savetxt(OUT_DATA, np.column_stack((S_fin, T_fin, U_fin, q_res)), header="S T U q1 q2 q3", comments='')
    print(f"\n已保存: {OUT_DATA}")

    # Plot
    OUT_PNG = f"STU_StrictS_{base}.png"
    plot_dense_ticks(dates, S_fin, T_fin, U_fin, q_res[:,2], OUT_PNG)

if __name__ == "__main__":
    main()