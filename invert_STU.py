#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STU 潮汐反演（双向融合 + 高斯滑窗 + 坐标轴科学计数法 + 更稳的“平滑正则”替代最小步长）
--------------------------------------------------------------------------------
核心思想（更稳）：
- 不在 L-BFGS-B 内部“限制最小步长”（SciPy 接口并不直接支持）；[web:1]
- 而是在每个滑窗反演的目标函数里加入时间平滑正则项：惩罚参数相对上一时刻 prev_x 的跳变；
- 仍保留你原有的 log-信任域（TRUST_RADIUS）来限制单次搜索范围。
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

# ================= 用户参数 =================
HALF_WINDOW = 3
COARSE_TRIALS = 30
COARSE_POINTS = 20
TRUST_RADIUS = 0.1

### NEW: 平滑正则强度（建议从小到大试）
# 这是“相对上一时刻”的惩罚权重，越大越平滑、但越可能压制真实变化。
SMOOTH_LAMBDA = 0.05

### NEW: 平滑正则在 log10 空间的尺度（更适合跨数量级参数）
# 惩罚项：lambda * sum( ((log10(x)-log10(prev_x))/smooth_scale)^2 )
SMOOTH_SCALE_LOG10 = np.array([0.15, 0.15, 0.20])  # [S,T,U] 可按经验微调

# ================= 物理模型 =================
rc = 0.171 / 2
rw = 0.150 / 2
bp = 26.26
b  = 40.44
S_S = 1e-5
zi = -29.8
wm2 = 2*np.pi/0.5175/86400

AQUIFER_CONFIG = {
    1: dict(S=(1e-5,1e-3),  T=(1e-4,1e-2),  U=(1e-7,1e-5)),
    2: dict(S=(5e-6,5e-4),  T=(1e-6,1e-3),  U=(1e-8,1e-6)),
    3: dict(S=(1e-6,1e-4),  T=(1e-7,1e-5),  U=(1e-9,1e-7)),
    4: dict(S=(1e-6,1e-4),  T=(1e-8,1e-4),  U=(1e-9,1e-6)),
    5: dict(S=(1e-6,5e-4),  T=(1e-9,1e-6),  U=(1e-11,1e-8)),
}
COARSE_BOUNDS = dict(S=(1e-6, 1e-4), T=(1e-6, 1e-4), U=(1e-10, 1e-7))


def select_data_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择数据文件")
    if not file_path:
        raise FileNotFoundError("未选择文件")
    return file_path


def forward(x, AR, pin):
    S, T, U = x
    if S <= 0 or T <= 0 or U <= 0:
        return np.array([1e6, 1e6, 1e6])

    Ss = S / b
    RKuB = (S * bp) / (b * S_S)
    K_K = U * bp
    D_D = U * (bp**2) / S_S

    delta = np.sqrt(2 * D_D / wm2)
    A = 1j * wm2 * S / T
    B = K_K / T
    C = (1 + 1j) / delta
    D_val = 1 / np.tanh(C * zi)
    belta = np.sqrt(A - B * C * D_val)

    E = 1j * wm2 * rc**2 * kv(0, belta * rw)
    F = 2 * T * belta * rw * kv(1, belta * rw)
    sigma = 1 + E / F

    N = 1 - np.cosh(C * zi)
    L = np.sinh(C * zi)

    H = S * 1j * wm2 + K_K * C * RKuB * (N / L)
    M = sigma * (S * 1j * wm2 - K_K * C * (1 / np.tanh(C * zi)))
    hw = H / M

    amp = np.abs(hw)
    pha = np.angle(hw) * 180 / np.pi

    q1 = amp - AR * Ss
    q2 = pha - pin
    return np.array([q1, q2, q1 + q2])


def misfit_masked(x, AR_chunk, pin_chunk, time_weights):
    S, T, U = x
    if S <= 0 or T <= 0 or U <= 0:
        return 1e99

    n = len(AR_chunk)
    err = 0.0
    weight_sum = 0.0
    for i in range(n):
        if time_weights[i] <= 1e-6:
            continue
        q1, q2, _ = forward(x, AR_chunk[i], pin_chunk[i])
        w = time_weights[i]
        err += (q1**2 + q2**2) * w
        weight_sum += w

    if weight_sum == 0:
        return 1e99
    return err / weight_sum


### NEW: 平滑正则（log10 空间）
def smooth_penalty_log10(x, prev_x, lam=SMOOTH_LAMBDA, scale_log10=SMOOTH_SCALE_LOG10):
    if prev_x is None:
        return 0.0
    x = np.asarray(x, float)
    prev_x = np.asarray(prev_x, float)
    if np.any(x <= 0) or np.any(prev_x <= 0):
        return 0.0
    d = (np.log10(x) - np.log10(prev_x)) / scale_log10
    return lam * float(np.sum(d**2))


def optimize_step_scipy(AR_chunk, pin_chunk, x0, cfg, time_weights, prev_x=None):
    """
    使用 L-BFGS-B 在 log 空间优化，但目标函数 = 数据失配 + 平滑正则项
    SciPy 的 L-BFGS-B 选项本身不提供“最小步长”控制，主要通过收敛阈值/线搜索运行。[web:1]
    """
    def obj_log(log_x):
        x = 10**log_x
        data_term = misfit_masked(x, AR_chunk, pin_chunk, time_weights)
        reg_term  = smooth_penalty_log10(x, prev_x)
        return data_term + reg_term

    log_x0 = np.log10(x0)

    # 信任域（你原来的逻辑）
    bounds = []
    for k, lx in zip(["S", "T", "U"], log_x0):
        gmin, gmax = np.log10(cfg[k])
        bounds.append((max(gmin, lx - TRUST_RADIUS), min(gmax, lx + TRUST_RADIUS)))

    # 选项：maxls/ftol/gtol 等是可控项，但不是“最小步长” [web:1]
    res = minimize(
        obj_log,
        log_x0,
        method='L-BFGS-B',
        bounds=bounds,
        options=dict(maxiter=80, maxls=30, ftol=1e-10, gtol=1e-08)
    )
    return 10**res.x


def optimize_init_optuna(AR_chunk, pin_chunk, cfg):
    Smin, Smax = cfg["S"]
    Tmin, Tmax = cfg["T"]
    Umin, Umax = cfg["U"]

    def objective(trial):
        S = trial.suggest_float("S", Smin, Smax, log=True)
        T = trial.suggest_float("T", Tmin, Tmax, log=True)
        U = trial.suggest_float("U", Umin, Umax, log=True)
        q1, q2, _ = forward([S, T, U], AR_chunk[0], pin_chunk[0])
        return q1**2 + q2**2

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=COARSE_TRIALS, show_progress_bar=False)
    bp0 = study.best_params
    return np.array([bp0["S"], bp0["T"], bp0["U"]])


def run_directional_pass(AR, pin, dates, cfg, init_guess=None):
    n = len(AR)
    S_res = np.zeros(n)
    T_res = np.zeros(n)
    U_res = np.zeros(n)

    prev_x = init_guess

    for i in range(n):
        idx_start = max(0, i - HALF_WINDOW)
        idx_end   = min(n, i + HALF_WINDOW + 1)

        AR_c   = AR[idx_start:idx_end]
        pin_c  = pin[idx_start:idx_end]
        dates_c = dates[idx_start:idx_end]

        curr_len = len(AR_c)
        rel_center = i - idx_start
        idxs = np.arange(curr_len)
        sigma = max(1.0, curr_len / 4.0)
        weights = np.exp(-0.5 * ((idxs - rel_center) / sigma) ** 2)

        center_date = dates[i]
        for k in range(curr_len):
            day_diff = abs((dates_c[k] - center_date).days)
            if day_diff > (HALF_WINDOW * 1.5):
                weights[k] = 0.0

        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(curr_len) / curr_len

        if prev_x is None:
            best_x = optimize_init_optuna([AR[i]], [pin[i]], cfg)
        else:
            best_x = optimize_step_scipy(AR_c, pin_c, prev_x, cfg, weights, prev_x=prev_x)

        S_res[i], T_res[i], U_res[i] = best_x
        prev_x = best_x

    return S_res, T_res, U_res


def auto_select_aquifer_type(AR, pin):
    print("\n>>> 自动判断含水层类型...")
    m = min(len(AR), COARSE_POINTS)
    S_l, T_l, U_l = [], [], []
    for i in range(m):
        res = optimize_init_optuna(
            [AR[i]], [pin[i]],
            dict(S=COARSE_BOUNDS["S"], T=COARSE_BOUNDS["T"], U=COARSE_BOUNDS["U"])
        )
        S_l.append(res[0])
        T_l.append(res[1])
        U_l.append(res[2])

    S_med = np.median(np.log10(S_l))
    T_med = np.median(np.log10(T_l))
    U_med = np.median(np.log10(U_l))

    best_t, best_d = None, 1e99
    for k, cfg in AQUIFER_CONFIG.items():
        Sm = np.mean(np.log10(cfg["S"]))
        Tm = np.mean(np.log10(cfg["T"]))
        Um = np.mean(np.log10(cfg["U"]))
        d = abs(S_med - Sm) + abs(T_med - Tm) + abs(U_med - Um)
        if d < best_d:
            best_d = d
            best_t = k

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

    fig.suptitle(f'Inversion Results (Aquifer Type {aq_type})', fontsize=14, y=0.92)
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
            if len(p) < 5:
                continue
            dates.append(datetime.strptime(p[0], "%Y/%m/%d"))
            HA.append(float(p[1]))
            HP.append(float(p[2]))
            EA.append(float(p[3]))
            EP.append(float(p[4]))

    dates = np.array(dates)
    n = len(dates)

    AR = (np.array(HA) / np.array(EA)) * 1e9
    pin = np.array(HP) - np.array(EP)

    aq_type = auto_select_aquifer_type(AR, pin)
    cfg = AQUIFER_CONFIG[aq_type]

    print("\n>>> 1/3 Forward Pass...")
    Sf, Tf, Uf = run_directional_pass(AR, pin, dates, cfg, init_guess=None)

    print(">>> 2/3 Backward Pass...")
    AR_rev, pin_rev, dates_rev = AR[::-1], pin[::-1], dates[::-1]
    init_rev = [Sf[-1], Tf[-1], Uf[-1]]
    Sb_r, Tb_r, Ub_r = run_directional_pass(AR_rev, pin_rev, dates_rev, cfg, init_guess=init_rev)
    Sb, Tb, Ub = Sb_r[::-1], Tb_r[::-1], Ub_r[::-1]

    print(">>> 3/3 Fusion...")
    w_fwd = np.linspace(0, 1, n)
    w_bwd = 1.0 - w_fwd
    S_fin = 10**(np.log10(Sf) * w_fwd + np.log10(Sb) * w_bwd)
    T_fin = 10**(np.log10(Tf) * w_fwd + np.log10(Tb) * w_bwd)
    U_fin = 10**(np.log10(Uf) * w_fwd + np.log10(Ub) * w_bwd)

    q_res = np.zeros((n, 3))
    for i in range(n):
        q1, q2, q3 = forward([S_fin[i], T_fin[i], U_fin[i]], AR[i], pin[i])
        q_res[i] = [q1, q2, q3]

    OUT_DATA = f"STU_SmoothReg_{base}.dat"
    np.savetxt(OUT_DATA, np.column_stack((S_fin, T_fin, U_fin, q_res)), header="S T U q1 q2 q3", comments='')
    print(f"\n已保存: {OUT_DATA}")

    OUT_PNG = f"STU_SmoothReg_{base}.png"
    plot_robust_labels(dates, S_fin, T_fin, U_fin, q_res[:, 2], aq_type, OUT_PNG)


if __name__ == "__main__":
    main()
