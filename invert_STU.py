#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STU 反演 - 方法三：粒子群优化 (PSO) [最终修复版]
------------------------------------------------
功能：
1. 使用粒子群算法 (PSO) 进行全局搜索，不依赖梯度，不易陷入局部最优。
2. 包含 robust (健壮) 的文件读取模块，解决日期格式错误。
3. 包含自适应平滑模块，解决数据量少时的报错。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import kv
from scipy.signal import savgol_filter
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog
import warnings

# 忽略数值计算中的溢出警告
warnings.filterwarnings("ignore")

# 全局绘图设置
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 用户参数设置 =================

# 文件路径 (建议修改为你实际的路径)
DATA_FILE_PATH = r"E:\临时\STU_VERSION-----\amp_pha_111111111.dat" 

# PSO 算法参数
PARTICLE_COUNT = 40    # 粒子数量 (越多越准，但越慢)
MAX_ITER = 60          # 每个时间点的迭代次数
W = 0.5                # 惯性权重 (保持飞行速度)
C1 = 1.5               # 个体学习因子 (向自己历史最好学)
C2 = 1.5               # 群体学习因子 (向群体最好学)

# 搜索范围 (Log10) - 在这里锁定 S 的量级
BOUNDS = {
    "S": [-5.5, -3.5], # S 限制在 1e-5.5 ~ 1e-3.5
    "T": [-7.0, -2.0], # T 宽范围
    "U": [-10.0, -5.0] # U 宽范围
}

# 权重因子 (抑制 S 波动)
WEIGHT_AMP = 10.0  # 重视振幅
WEIGHT_PHA = 0.1   # 轻视相位

# ================= 2. 物理模型 =================
rc=0.171/2; rw=0.150/2; bp=26.26; b=40.44; S_S=1e-5; zi=-29.8; wm2=2*np.pi/0.5175/86400

def forward(x, AR, pin):
    """
    物理前向模型
    """
    S, T, U = x
    # 物理约束：防止负数或极小值
    if S <= 1e-15 or T <= 1e-15 or U <= 1e-15: 
        return 1e6, 1e6
    
    try:
        Ss = S/b
        RKuB = (S*bp)/(b*S_S)
        K_K = U*bp
        D_D = U*(bp**2)/S_S
        
        delta = np.sqrt(2*D_D/wm2)
        A = 1j*wm2*S/T
        B = K_K/T
        
        C = (1+1j)/delta
        C_zi = C*zi
        
        # 数值稳定处理
        if abs(C_zi) > 50: 
            D_val = 1.0
            term_NL = -1.0
        else: 
            D_val = 1.0/np.tanh(C_zi)
            term_NL = -np.tanh(C_zi/2.0)
        
        belta = np.sqrt(A - B*C*D_val)
        E = 1j*wm2*rc**2 * kv(0, belta*rw)
        F = 2*T*belta*rw * kv(1, belta*rw)
        
        sigma = 1 + E/F
        H = S*1j*wm2 + K_K*C*RKuB*term_NL
        M = sigma*(S*1j*wm2 - K_K*C*D_val)
        
        hw = H/M
        
        q1 = np.abs(hw) - AR*Ss
        q2 = (np.angle(hw)*180/np.pi) - pin
        return q1, q2
    except:
        return 1e6, 1e6

# ================= 3. PSO 核心算法类 =================
class PSO_Solver:
    def __init__(self, AR_val, pin_val):
        self.AR = AR_val
        self.pin = pin_val
        self.dim = 3 # S, T, U
        
        # 定义搜索边界
        self.lb = np.array([BOUNDS["S"][0], BOUNDS["T"][0], BOUNDS["U"][0]])
        self.ub = np.array([BOUNDS["S"][1], BOUNDS["T"][1], BOUNDS["U"][1]])
        
        # 初始化粒子群 (在 Log 空间均匀分布)
        self.X = np.random.uniform(self.lb, self.ub, (PARTICLE_COUNT, self.dim))
        self.V = np.random.uniform(-0.1, 0.1, (PARTICLE_COUNT, self.dim))
        
        # 记录最优解
        self.pbest_X = self.X.copy()
        self.pbest_score = np.full(PARTICLE_COUNT, np.inf)
        self.gbest_X = np.zeros(self.dim)
        self.gbest_score = np.inf
        
    def calculate_fitness(self, x_log):
        """计算适应度 (Misfit)"""
        params = 10**x_log # 还原为线性值
        q1, q2 = forward(params, self.AR, self.pin)
        
        # 加权 Misfit: 压制相位权重，防止 S 乱跳
        cost = (q1 * WEIGHT_AMP)**2 + (q2 * WEIGHT_PHA)**2
        return cost

    def run(self):
        """执行迭代"""
        for i in range(MAX_ITER):
            # 1. 计算适应度
            for j in range(PARTICLE_COUNT):
                score = self.calculate_fitness(self.X[j])
                
                # 更新个体最优
                if score < self.pbest_score[j]:
                    self.pbest_score[j] = score
                    self.pbest_X[j] = self.X[j].copy()
                
                # 更新群体最优
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_X = self.X[j].copy()
            
            # 2. 更新速度和位置
            r1 = np.random.rand(PARTICLE_COUNT, self.dim)
            r2 = np.random.rand(PARTICLE_COUNT, self.dim)
            
            self.V = W * self.V + \
                     C1 * r1 * (self.pbest_X - self.X) + \
                     C2 * r2 * (self.gbest_X - self.X)
            
            self.X = self.X + self.V
            
            # 3. 边界限制 (Clip)
            self.X = np.clip(self.X, self.lb, self.ub)
            
        return 10**self.gbest_X, self.gbest_score

# ================= 4. 主程序 =================
def select_file_gui():
    """如果没有找到默认文件，弹出选择框"""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="选择数据文件 (*.dat)")
    return path

def main():
    # 1. 确定文件路径
    file_path = DATA_FILE_PATH
    if not os.path.exists(file_path):
        print(f"❌ 默认路径未找到: {file_path}")
        print(">>> 请在弹窗中选择文件...")
        file_path = select_file_gui()
        if not file_path:
            print("未选择文件，程序退出。")
            return

    # 2. 读取数据 (Robust Reading)
    dates, AR, pin = [], [], []
    print(f"正在读取文件: {os.path.basename(file_path)} ...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        line_num = 0
        for line in f:
            line_num += 1
            p = line.split()
            if len(p) < 5: continue
            
            try:
                # 尝试解析日期 (兼容 / 和 -)
                date_str = p[0]
                try:
                    d = datetime.strptime(date_str, "%Y/%m/%d")
                except ValueError:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                
                # 读取数值
                ha, hp, ea, ep = map(float, p[1:5])
                
                # 简单清洗
                if abs(ea) < 1e-9: continue # 除0保护
                
                ar_val = (ha / ea) * 1e9
                pin_val = hp - ep
                
                dates.append(d)
                AR.append(ar_val)
                pin.append(pin_val)
            except:
                # 忽略解析错误的行（如表头）
                continue
    
    n = len(dates)
    if n == 0:
        print("❌ 错误：文件中没有读取到有效数据！请检查日期格式。")
        return
    
    print(f"✅ 成功读取 {n} 个数据点。")
    print(f">>> 开始 PSO 全局搜索 (Particles={PARTICLE_COUNT}, Iter={MAX_ITER})...")
    
    # 3. 执行反演
    S_res, T_res, U_res, Q_res = [], [], [], []
    
    for i in range(n):
        solver = PSO_Solver(AR[i], pin[i])
        best_stu, best_misfit = solver.run()
        
        S_res.append(best_stu[0])
        T_res.append(best_stu[1])
        U_res.append(best_stu[2])
        Q_res.append(best_misfit)
        
        if (i+1) % 10 == 0 or i == n-1:
            print(f"    进度: {i+1}/{n} | Misfit: {best_misfit:.4f}")

    # 转为 numpy 数组
    S_res = np.array(S_res)
    T_res = np.array(T_res)
    U_res = np.array(U_res)
    
    # 4. 后处理平滑 (解决 PSO 抖动 + 修复报错)
    print(">>> 正在进行结果平滑...")
    
    # 自动计算安全窗口大小
    # 必须是奇数，且小于数据长度
    target_window = 15
    safe_window = target_window
    if n < target_window:
        safe_window = n if n % 2 == 1 else n - 1
    
    # 只有当数据点足够多时才平滑
    if safe_window >= 3:
        try:
            S_smooth = savgol_filter(S_res, window_length=safe_window, polyorder=2)
            T_smooth = savgol_filter(T_res, window_length=safe_window, polyorder=2)
            U_smooth = savgol_filter(U_res, window_length=safe_window, polyorder=2)
        except Exception as e:
            print(f"⚠️ 平滑失败 ({e})，显示原始 PSO 结果。")
            S_smooth, T_smooth, U_smooth = S_res, T_res, U_res
    else:
        print("⚠️ 数据点过少，跳过平滑步骤。")
        S_smooth, T_smooth, U_smooth = S_res, T_res, U_res

    # 5. 保存结果
    out_file = file_path.replace(".dat", "_PSO_Result.dat")
    np.savetxt(out_file, np.column_stack((S_smooth, T_smooth, U_smooth, Q_res)), 
               header="S T U Misfit", comments='')
    print(f"✅ 数据已保存: {out_file}")

    # 6. 绘图
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # S
    axes[0].plot(dates, S_res, '.', color='gray', alpha=0.3, label='Raw PSO')
    axes[0].plot(dates, S_smooth, 'r-', lw=2, label='Smoothed')
    axes[0].set_ylabel('S (Storage)', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    
    # T
    axes[1].plot(dates, T_smooth, 'b-', lw=1.5)
    axes[1].set_ylabel('T (Transmissivity)', fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    
    # U
    axes[2].plot(dates, U_smooth, 'g-', lw=1.5)
    axes[2].set_ylabel('U (Leakage)', fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, which='both', ls='--', alpha=0.5)
    
    # Misfit
    axes[3].plot(dates, Q_res, 'k-', lw=1)
    axes[3].set_ylabel('Weighted Misfit', fontweight='bold')
    axes[3].grid(True, which='major', ls='-', alpha=0.5)
    
    # 时间轴格式
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.suptitle(f"PSO Global Inversion: {os.path.basename(file_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()