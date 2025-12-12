#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STU 潮汐反演（S 波动抑制版 + 固定文件路径）
----------------------------------------------------
修改点：
1. 【相位降权】：misfit 中降低相位权重，防止 S 因相位噪声跳动。
2. 【严格限速】：TRUST_RADIUS = 0.01，强制连续。
3. 【固定文件】：直接指定文件路径，不再弹窗。
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
import warnings

warnings.filterwarnings("ignore")

# 全局绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 用户参数 =================

# 0. 固定文件路径 (请修改为你自己的路径)
# 注意：Windows路径建议用双反斜杠 \\ 或反斜杠 /
DATA_FILE_PATH = r"E:\临时\STU_VERSION-----\amp_pha_111111111.dat"  # <--- 修改这里！

# 1. 滑窗半径 (适当增大以平滑)
HALF_WINDOW = 5   

# 2. 信任域 (限速器)
# 0.01 表示相邻点变化不能超过 2.3%，这能有效消除锯齿
TRUST_RADIUS = 0.01

# 3. 权重因子 (关键修改)
WEIGHT_AMP = 10.0  # 重视振幅 (控制 T)
WEIGHT_PHA = 0.1   # 轻视相位 (控制 S，防止乱跳)

# 4. 自动分类配置
COARSE_TRIALS = 30
COARSE_POINTS = 20

# ================= 物理模型 =================
rc=0.171/2; rw=0.150/2; bp=26.26; b=40.44; S_S=1e-5; zi=-29.8; wm2=2*np.pi/0.5175/86400
AQUIFER_CONFIG = {
    1: dict(S=(1e-5,1e-3),  T=(1e-4,1e-2),  U=(1e-7,1e-5)),
    2: dict(S=(5e-6,5e-4),  T=(1e-6,1e-3),  U=(1e-8,1e-6)),
    3: dict(S=(1e-6,1e-4),  T=(1e-7,1e-5),  U=(1e-9,1e-7)),
    4: dict(S=(1e-6,1e-4),  T=(1e-8,1e-4),  U=(1e-9,1e-6)),
    5: dict(S=(1e-6,5e-4),  T=(1e-9,1e-6),  U=(1e-11,1e-8)),
}
COARSE_BOUNDS = dict(S=(1e-6, 1e-4), T=(1e-6, 1e-4), U=(1e-10, 1e-7))

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

# ============ 修改后的 misfit (带权重 + 忽略断点) ============
def misfit_weighted_masked(x, AR_chunk, pin_chunk, time_weights):
    S, T, U = x
    if S <= 0 or T <= 0 or U <= 0: return 1e99
    
    n = len(AR_chunk)
    err = 0.0
    weight_sum = 0.0
    
    for i in range(n):
        if time_weights[i] <= 1e-6: continue
        
        q1, q2, _ = forward(x, AR_chunk[i], pin_chunk[i])
        w = time_weights[i]
        
        # 【关键修改】分别加权
        term_amp = (q1 * WEIGHT_AMP) ** 2
        term_pha = (q2 * WEIGHT_PHA) ** 2
        
        err += (term_amp + term_pha) * w
        weight_sum += w
        
    if weight_sum == 0: return 1e99
    return err / weight_sum

def optimize_step_scipy(AR_chunk, pin_chunk, x0, cfg, time_weights):
    # 使用带权重的 misfit
    def obj_log(log_x): 
        return misfit_weighted_masked(10**log_x, AR_chunk, pin_chunk, time_weights)
    
    log_x0 = np.log10(x0)
    bounds = []
    for k, lx in zip(["S","T","U"], log_x0):
        gmin, gmax = np.log10(cfg[k])
        # 信任域限制 (TRUST_RADIUS)
        bounds.append((max(gmin, lx - TRUST_RADIUS), min(gmax, lx + TRUST_RADIUS)))
    res = minimize(obj_log, log_x0, method='L-BFGS-B', bounds=bounds)
    return 10**res.x

def optimize_init_optuna(AR_chunk, pin_chunk, cfg):
    Smin, Smax = cfg["S"]; Tmin, Tmax = cfg["T"]; Umin, Umax = cfg["U"]
    def objective(trial):
        S = trial.suggest_float("S", Smin, Smax, log=True)
        T = trial.suggest_float("T", Tmin, Tmax, log=True)
        U = trial.suggest_float("U", Umin, Umax, log=True)
        # 初始化也用简单的加权 misfit (无时间权重)
        q1, q2, _ = forward([S,T,U], AR_chunk[0], pin_chunk[0])
        err = (q1*WEIGHT_AMP)**2 + (q2*WEIGHT_PHA)**2
        return err
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    bp = study.best_params
    return np.array([bp["S"], bp["T"], bp["U"]])

def run_directional_pass(AR, pin, dates, cfg, init_guess=None):
    n = len(AR)
    S_res = np.zeros(n); T_res = np.zeros(n); U_res = np.zeros(n)
    prev_x = init_guess
    for i in range(n):
        idx_start = max(0, i - HALF_WINDOW)
        idx_end   = min(n, i + HALF_WINDOW + 1)
        AR_c  = AR[idx_start:idx_end]; pin_c = pin[idx_start:idx_end]; dates_c = dates[idx_start:idx_end]
        
        curr_len = len(AR_c); rel_center = i - idx_start
        idxs = np.arange(curr_len)
        sigma = max(1.0, curr_len / 4.0)
        weights = np.exp(-0.5 * ((idxs - rel_center) / sigma) ** 2)
        
        center_date = dates[i]
        for k in range(curr_len):
            day_diff = abs((dates_c[k] - center_date).days)
            if day_diff > (HALF_WINDOW * 1.5): weights[k] = 0.0
                
        if weights.sum() > 0: weights /= weights.sum()
        else: weights = np.ones(curr_len)/curr_len
        
        if prev_x is None: best_x = optimize_init_optuna(AR_c, pin_c, cfg)
        else: best_x = optimize_step_scipy(AR_c, pin_c, prev_x, cfg, weights)
            
        S_res[i], T_res[i], U_res[i] = best_x
        prev_x = best_x
    return S_res, T_res, U_res

def auto_select_aquifer_type(AR, pin):
    print("\n>>> 自动判断含水层类型...")
    m = min(len(AR), COARSE_POINTS)
    S_l, T_l, U_l = [], [], []
    for i in range(m):
        # 这里的粗反演不加权，只做量级判断
        def obj(trial):
            S = trial.suggest_float("S", *COARSE_BOUNDS["S"], log=True)
            T = trial.suggest_float("T", *COARSE_BOUNDS["T"], log=True)
            U = trial.suggest_float("U", *COARSE_BOUNDS["U"], log=True)
            q1, q2, _ = forward([S,T,U], AR[i], pin[i])
            return q1**2 + q2**2
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(obj, n_trials=15, show_progress_bar=False)
        bp = study.best_params
        S_l.append(bp["S"]); T_l.append(bp["T"]); U_l.append(bp["U"])
        
    S_med = np.median(np.log10(S_l)); T_med = np.median(np.log10(T_l)); U_med = np.median(np.log10(U_l))
    best_t, best_d = None, 1e99
    for k, cfg in AQUIFER_CONFIG.items():
        Sm = np.mean(np.log10(cfg["S"])); Tm = np.mean(np.log10(cfg["T"])); Um = np.mean(np.log10(cfg["U"]))
        d = abs(S_med - Sm) + abs(T_med - Tm) + abs(U_med - Um)
        if d < best_d: best_d = d; best_t = k
    print(f"    判定类型: {best_t}")
    return best_t

def plot_robust_labels(dates, S, T, U, Misfit, aq_type, filename):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.1) 
    data_list = [S, T, U]
    labels = [r'$S$ (Storage)', r'$T$ (Transmissivity)', r'$U$ (Leakage)']
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    for i in range(3):
        ax = axes[i]
        ax.plot(dates, data_list[i], color=colors[i], linewidth=1.5, label=labels[i])
        ax.set_ylabel(labels[i].split()[0], fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        maj_loc = ticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=15)
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
    ax4.grid(True, which='major', linestyle='-', alpha=0.6)
    ax4.tick_params(which='both', direction='in', top=True, right=True)
    ax4.legend(loc='upper right')

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%Y-%m')
    ax4.xaxis.set_major_locator(locator)
    ax4.xaxis.set_major_formatter(formatter)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Inversion Results (Type {aq_type}, Weighted)', fontsize=14, y=0.92)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已绘图: {filename}")
    plt.show()

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(DATA_FILE_PATH):
        print(f"错误: 找不到文件 {DATA_FILE_PATH}")
        print("请在代码开头 DATA_FILE_PATH 处修改路径。")
        return
        
    base = os.path.splitext(os.path.basename(DATA_FILE_PATH))[0]
    print(f"处理文件: {base}")
    
    dates, HA, HP, EA, EP = [], [], [], [], []
    with open(DATA_FILE_PATH, "r") as f:
        for line in f:
            p = line.split()
            if len(p)<5: continue
            try:
                d = datetime.strptime(p[0], "%Y/%m/%d")
                ha, hp, ea, ep = map(float, p[1:5])
                dates.append(d); HA.append(ha); HP.append(hp); EA.append(ea); EP.append(ep)
            except: pass
            
    dates = np.array(dates); n = len(dates)
    AR = (np.array(HA)/np.array(EA)) * 1e9
    pin = np.array(HP) - np.array(EP)

    aq_type = auto_select_aquifer_type(AR, pin)
    cfg = AQUIFER_CONFIG[aq_type]

    # 1. Forward
    print("\n>>> 1/3 Forward Pass...")
    Sf, Tf, Uf = run_directional_pass(AR, pin, dates, cfg, init_guess=None)
    
    # 2. Backward
    print(">>> 2/3 Backward Pass...")
    AR_rev, pin_rev, dates_rev = AR[::-1], pin[::-1], dates[::-1]
    init_rev = [Sf[-1], Tf[-1], Uf[-1]]
    Sb_r, Tb_r, Ub_r = run_directional_pass(AR_rev, pin_rev, dates_rev, cfg, init_guess=init_rev)
    Sb, Tb, Ub = Sb_r[::-1], Tb_r[::-1], Ub_r[::-1]

    # 3. Fusion
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

    OUT_DATA = f"STU_Weighted_{base}.dat"
    np.savetxt(OUT_DATA, np.column_stack((S_fin, T_fin, U_fin, q_res)), header="S T U q1 q2 q3", comments='')
    print(f"\n已保存: {OUT_DATA}")

    # Plot
    OUT_PNG = f"STU_Weighted_{base}.png"
    plot_robust_labels(dates, S_fin, T_fin, U_fin, q_res[:,2], aq_type, OUT_PNG)

if __name__ == "__main__":
    main()