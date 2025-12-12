#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STU 潮汐反演（全局正则化反演 Global Regularized Inversion）
----------------------------------------------------
核心升级：
1. 【全局反演】：不再使用滑窗，而是将所有时间点作为一个整体进行优化。
2. 【平滑约束】：在目标函数中加入一阶粗糙度惩罚 (Roughness Penalty)。
3. 【ABIC思想】：通过调整平滑权重 (ALPHA)，在拟合差和曲线光滑度之间找到最佳平衡。

这是解决 S 波动大、不连续问题的终极数学方案。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
from scipy.special import kv
from scipy.optimize import minimize
import os
import warnings

warnings.filterwarnings("ignore")

# 全局绘图参数
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 用户参数 (ABIC 调优区) =================

# 固定文件路径
DATA_FILE_PATH = r"E:\临时\STU_VERSION-----\amp_pha_111111111.dat"

# 1. 平滑权重 (Alpha) - ABIC 的核心参数
# 值越大，曲线越光滑（越接近直线），但拟合误差会增加。
# 值越小，曲线越抖动（越贴近数据），但噪声会增加。
# 建议尝试：0.1, 1.0, 10.0, 50.0
ALPHA_S = 50.0   # S 的平滑权重 (给大一点，按住 S)
ALPHA_T = 5.0    # T 的平滑权重
ALPHA_U = 1.0    # U 的平滑权重

# 2. 权重因子 (数据拟合)
WEIGHT_AMP = 10.0  # 振幅重要性
WEIGHT_PHA = 0.1   # 相位重要性 (降权以防止 S 跳动)

# 3. 物理约束范围 (Log10)
# 可以在这里锁死 S 的量级
BOUNDS_LOG = {
    "S": (-5.0, -4.0),  # S 限制在 1e-5 ~ 1e-4
    "T": (-7.0, -2.0),  # T 宽范围
    "U": (-10.0, -5.0)  # U 宽范围
}

# ================= 物理模型 =================
rc=0.171/2; rw=0.150/2; bp=26.26; b=40.44; S_S=1e-5; zi=-29.8; wm2=2*np.pi/0.5175/86400

def forward(x, AR, pin):
    S, T, U = x
    if S <= 1e-15 or T <= 1e-15 or U <= 1e-15: return np.array([1e6, 1e6])
    
    # 物理方程
    Ss, RKuB = S/b, (S*bp)/(b*S_S); K_K, D_D = U*bp, U*(bp**2)/S_S
    delta = np.sqrt(2*D_D/wm2); A = 1j*wm2*S/T; B = K_K/T
    C = (1+1j)/delta
    
    # 数值稳定性处理 (tanh)
    C_zi = C*zi
    if abs(C_zi) > 50: 
        D_val = 1.0; term_NL = -1.0 # 极限情况
    else:
        D_val = 1.0/np.tanh(C_zi); term_NL = -np.tanh(C_zi/2.0)
        
    belta = np.sqrt(A - B*C*D_val)
    E = 1j*wm2*rc**2 * kv(0, belta*rw); F = 2*T*belta*rw * kv(1, belta*rw)
    sigma = 1 + E/F
    H = S*1j*wm2 + K_K*C*RKuB*term_NL; M = sigma*(S*1j*wm2 - K_K*C*D_val)
    hw = H/M
    
    q1 = np.abs(hw) - AR*(S/b)
    q2 = (np.angle(hw)*180/np.pi) - pin
    return np.array([q1, q2])

# ============ 全局目标函数 (Global Objective Function) ============
def global_objective(params_flat, AR_all, pin_all, n_points):
    """
    params_flat: 所有天数的参数铺平 [S0..Sn, T0..Tn, U0..Un] (log10 space)
    """
    # 还原参数矩阵
    S_logs = params_flat[0:n_points]
    T_logs = params_flat[n_points:2*n_points]
    U_logs = params_flat[2*n_points:]
    
    S_vals = 10**S_logs
    T_vals = 10**T_logs
    U_vals = 10**U_logs
    
    total_misfit = 0.0
    
    # 1. Data Misfit (数据拟合项)
    # 这里可以使用向量化加速，但为了物理清晰，写成循环
    for i in range(n_points):
        q1, q2 = forward([S_vals[i], T_vals[i], U_vals[i]], AR_all[i], pin_all[i])
        # 加权残差
        err = (q1 * WEIGHT_AMP)**2 + (q2 * WEIGHT_PHA)**2
        total_misfit += err
        
    # 2. Regularization (平滑约束项) - 这就是 ABIC 里的 Roughness
    # 计算相邻点的差分 (一阶导数)
    diff_S = np.diff(S_logs)
    diff_T = np.diff(T_logs)
    diff_U = np.diff(U_logs)
    
    reg_S = np.sum(diff_S**2) * (ALPHA_S**2)
    reg_T = np.sum(diff_T**2) * (ALPHA_T**2)
    reg_U = np.sum(diff_U**2) * (ALPHA_U**2)
    
    # 总目标 = 拟合误差 + 平滑惩罚
    total_cost = total_misfit + reg_S + reg_T + reg_U
    return total_cost

# ============ 辅助：生成初始猜测 ============
def generate_initial_guess(AR, pin, n):
    # 简单计算一个全局平均的最优解作为初值
    # 这样优化器只需要微调，速度快
    def obj_single(x):
        idx = n // 2
        q1, q2 = forward(10**x, AR[idx], pin[idx])
        return q1**2 + q2**2
    
    res = minimize(obj_single, [-4.5, -4.0, -8.0], method='L-BFGS-B', 
                   bounds=[BOUNDS_LOG["S"], BOUNDS_LOG["T"], BOUNDS_LOG["U"]])
    
    s0, t0, u0 = res.x
    print(f">>> 初始猜测基准: S={10**s0:.2e}, T={10**t0:.2e}, U={10**u0:.2e}")
    
    # 铺平成长向量
    init_S = np.full(n, s0)
    init_T = np.full(n, t0)
    init_U = np.full(n, u0)
    return np.concatenate([init_S, init_T, init_U])

# ============ 绘图函数 ============
def plot_results(dates, S, T, U, q_res, filename):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.1) 
    
    data = [S, T, U]
    names = ['S', 'T', 'U']
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(dates, data[i], color=colors[i], lw=2)
        ax.set_ylabel(names[i], fontweight='bold')
        ax.set_yscale('log')
        
        # 强制显示密集刻度
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=np.arange(1,10), numticks=15))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        ax.grid(True, which='major', ls='-', alpha=0.6)
        ax.grid(True, which='minor', ls=':', alpha=0.3)
        ax.tick_params(direction='in', which='both', right=True)

    # Misfit
    ax4 = axes[3]
    # 计算 q3 = q1 + q2 (近似)
    misfit_val = np.sqrt(q_res[:,0]**2 + q_res[:,1]**2)
    ax4.plot(dates, misfit_val, 'k-', lw=1)
    ax4.set_ylabel('Misfit', fontweight='bold')
    ax4.grid(True, ls='-', alpha=0.6)
    ax4.tick_params(direction='in', which='both', right=True)
    
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    fig.suptitle(f'Global Regularized Inversion (Alpha_S={ALPHA_S})', y=0.92)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成: {filename}")
    plt.show()

# ============ 主程序 ============
def main():
    if not os.path.exists(DATA_FILE_PATH):
        print("错误: 文件不存在，请修改路径")
        return
    
    print(f"读取文件: {DATA_FILE_PATH}")
    dates, AR, pin = [], [], []
    with open(DATA_FILE_PATH, 'r') as f:
        for line in f:
            p = line.split()
            if len(p)<5: continue
            try:
                d = datetime.strptime(p[0], "%Y/%m/%d")
                dates.append(d)
                # AR = (HA/EA)*1e9, pin = HP - EP
                AR.append(float(p[1])/float(p[3])*1e9)
                pin.append(float(p[2])-float(p[4]))
            except: pass
            
    dates = np.array(dates)
    AR = np.array(AR)
    pin = np.array(pin)
    n = len(dates)
    
    print(f"数据点数: {n}")
    
    # 1. 生成初值
    x0 = generate_initial_guess(AR, pin, n)
    
    # 2. 设置边界
    # 我们需要为每一个点的每一个参数都设置边界
    bounds = []
    # S bounds
    bounds.extend([BOUNDS_LOG["S"]] * n)
    # T bounds
    bounds.extend([BOUNDS_LOG["T"]] * n)
    # U bounds
    bounds.extend([BOUNDS_LOG["U"]] * n)
    
    # 3. 运行全局优化 (Global Optimization)
    print(">>> 开始全局正则化反演 (这可能需要几秒到一分钟)...")
    # L-BFGS-B 非常适合处理这种大规模边界约束问题
    res = minimize(
        global_objective, 
        x0, 
        args=(AR, pin, n), 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'disp': True, 'maxiter': 2000}
    )
    
    print(f">>> 反演完成! Success: {res.success}, Msg: {res.message}")
    
    # 4. 解析结果
    final_x = res.x
    S_fin = 10**final_x[0:n]
    T_fin = 10**final_x[n:2*n]
    U_fin = 10**final_x[2*n:]
    
    # 计算残差
    q_res = []
    for i in range(n):
        q = forward([S_fin[i], T_fin[i], U_fin[i]], AR[i], pin[i])
        q_res.append(q)
    q_res = np.array(q_res)
    
    # 5. 保存与绘图
    base = os.path.splitext(os.path.basename(DATA_FILE_PATH))[0]
    OUT_DATA = f"STU_GlobalABIC_{base}.dat"
    np.savetxt(OUT_DATA, np.column_stack((S_fin, T_fin, U_fin, q_res)), header="S T U q1 q2", comments='')
    print(f"数据已保存: {OUT_DATA}")
    
    OUT_PNG = f"STU_GlobalABIC_{base}.png"
    plot_results(dates, S_fin, T_fin, U_fin, q_res, OUT_PNG)

if __name__ == "__main__":
    main()