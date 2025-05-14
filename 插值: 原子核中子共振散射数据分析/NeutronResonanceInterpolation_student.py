import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 实验数据
energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
error = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])  # mb

def lagrange_interpolation(x, x_data, y_data):
    """
    实现拉格朗日多项式插值
    
    参数:
        x: 插值点或数组
        x_data: 已知数据点的x坐标
        y_data: 已知数据点的y坐标
        
    返回:
        插值结果
        
    提示:
        1. 使用拉格朗日插值公式实现
        2. 考虑使用双重循环结构
        3. 注意处理分母为零的情况
    """
    n = len(x_data)
    # 确保结果是浮点类型
    if isinstance(x, (int, float)):
        result = 0.0
    else:
        result = np.zeros_like(x, dtype=float)
    
    for i in range(n):
        if isinstance(x, (int, float)):
            term = 1.0
        else:
            term = np.ones_like(x, dtype=float)
            
        for j in range(n):
            if i != j:
                denominator = x_data[i] - x_data[j]
                if denominator == 0:
                    continue
                term *= (x - x_data[j]) / denominator
        result += y_data[i] * term
    
    return result

def cubic_spline_interpolation(x, x_data, y_data):
    """
    实现三次样条插值
    
    参数:
        x: 插值点或数组
        x_data: 已知数据点的x坐标
        y_data: 已知数据点的y坐标
        
    返回:
        插值结果
        
    提示:
        1. 使用scipy.interpolate.interp1d
        2. 设置kind='cubic'
        3. 考虑边界条件处理
    """
    f = interp1d(x_data, y_data, kind='cubic', fill_value='extrapolate')
    return f(x)

def find_peak(x, y):
    """
    寻找峰值位置和半高全宽(FWHM)
    
    参数:
        x: x坐标数组
        y: y坐标数组
        
    返回:
        tuple: (峰值位置, FWHM)
        
    提示:
        1. 使用np.argmax找到峰值位置
        2. 计算半高位置
        3. 使用np.argmin找到半高位置
    """
    # 找到峰值位置
    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    
    # 计算半高
    half_max = peak_y / 2
    
    # 找到左边半高点
    left_idx = np.argmin(np.abs(y[:peak_idx] - half_max))
    left_x = x[left_idx]
    
    # 找到右边半高点
    right_idx = peak_idx + np.argmin(np.abs(y[peak_idx:] - half_max))
    right_x = x[right_idx]
    
    # 计算FWHM
    fwhm = right_x - left_x
    
    return peak_x, fwhm

def plot_results():
    """
    绘制插值结果和原始数据对比图
    
    提示:
        1. 生成密集的插值点
        2. 调用前面实现的插值函数
        3. 绘制原始数据点和插值曲线
    """
    # 生成密集的插值点
    x_interp = np.linspace(0, 200, 500)
    
    # 计算两种插值结果
    lagrange_result = lagrange_interpolation(x_interp, energy, cross_section)
    spline_result = cubic_spline_interpolation(x_interp, energy, cross_section)
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    
    # 原始数据点
    plt.errorbar(energy, cross_section, yerr=error, fmt='o', color='black', 
                label='Original Data', capsize=5)
    
    # 插值曲线
    plt.plot(x_interp, lagrange_result, '-', label='Lagrange Interpolation')
    plt.plot(x_interp, spline_result, '--', label='Cubic Spline Interpolation')
    
    # 标记峰值
    lagrange_peak, lagrange_fwhm = find_peak(x_interp, lagrange_result)
    spline_peak, spline_fwhm = find_peak(x_interp, spline_result)
    
    plt.axvline(lagrange_peak, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 图表装饰
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title('Neutron Resonance Scattering Cross Section Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    plot_results()
