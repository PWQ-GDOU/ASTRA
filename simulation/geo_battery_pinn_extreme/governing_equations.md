# 控制方程

## 一、系统级退化 ODE（全局常微分方程）

### 状态变量
| 变量 | 含义 | 初始值 | 单位 |
|------|------|--------|------|
| `soh` | 健康状态 (State of Health) | 1.0 | - |
| `rint` | 等效内阻 | 0.034 | Ω |
| `tcell` | 电池温度 | 25 | °C |

### SOH 退化方程

$$
\frac{d(\text{soh})}{dt} = -k_{\text{deg}} \cdot \sigma
$$

其中退化应力 $sigma$ 为:

$$
\sigma = \text{clamp}\left((0.20 + 0.92 \cdot F_{\text{DOD}}^{1.35} + 0.45 \cdot a) \cdot \alpha_T \cdot f, \; 0, \; 5\right)
$$

### 内阻增长方程

$$
\frac{d(\text{rint})}{dt} = R_{\text{base}} \cdot 2.8 \cdot k_{\text{deg}} \cdot \sigma
$$

### 热动态方程

$$
\frac{d(\text{tcell})}{dt} = \frac{T_{\text{target}} - \text{tcell}}{18}
$$

## 二、三维固体传热方程

$$
\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q}_{\text{joule}}
$$

体热源:
$$
\dot{q}_{\text{joule}} = \frac{I_{\text{dis}}^2 \cdot r_{\text{int}} \cdot (0.36 + F_{\text{DOD}})}{V_{\text{cell}}}
$$

## 三、退化应力分量

### 放电深度特征
$$
F_{\text{DOD}} = 0.8 \cdot \frac{t_{\text{shadow}}}{72}
$$

### 地影时长模型（GEO轨道）
$$
t_{\text{shadow}} = 5 + 67 \cdot \sin(\pi \cdot \phi_{46})^{0.65} \quad [\text{min}]
$$

### 温度加速因子 (Arrhenius形式)
$$
\alpha_T = \exp\left(0.048 \cdot (\text{clamp}(T_{\text{cell}}, -20, 60) - 25)\right)
$$

### 化成周期效应
$$
f = 1 + 1.35 \cdot \exp(-N_{\text{cycle}} / 18)
$$

### 老化因子
$$
a = 1 - \text{soh}
$$

## 四、辅助方程

### 环境温度
$$
T_{\text{amb}} = 25 + 0.35 \cdot \sin(2\pi \cdot \phi_{46}) \quad [°C]
$$

### 热目标温度
$$
T_{\text{target}} = T_{\text{amb}} + 30 \cdot \dot{Q}_{\text{gen}}
$$

其中:
$$
\dot{Q}_{\text{gen}} = I_{\text{dis}}^2 \cdot r_{\text{int}} \cdot (0.36 + F_{\text{DOD}})
$$

### 容量
$$
C_{\text{cap}} = Q_{\text{mAh}} \cdot \text{soh} \quad [\text{mAh}]
$$

### 近似剩余寿命
$$
\text{RUL} = \max\left(\frac{\text{soh} - \text{SOH}_{\text{fail}}}{k_{\text{deg}} \cdot \sigma}, \; 0\right) \quad [\text{cycles}]
$$

## 五、PINN 电压响应特征

### 三指数电压降模型参数（极端段特征）

**快速分量 (fast):**
$$
A_f = 0.0075 + 0.058a + 0.0055F_{\text{DOD}} + 0.22 \cdot \max(r_{\text{int}} - R_{\text{base}}, 0)
$$
$$
\tau_f = 12 + 60a + 3.8F_{\text{DOD}}

**中速分量 (mid):**
$$
A_m = 0.0035 + 0.028a + 0.0022F_{\text{DOD}} + 0.00055 \cdot \max(T_{\text{cell}} - 25, 0)
$$
$$
\tau_m = 76 + 155a + 12F_{\text{DOD}}

**尾部分量 (tail):**
$$
A_t = 0.0012 + 0.018a^2 + 0.0015F_{\text{DOD}}
$$
$$
\tau_t = 240 + 260a
$$

### 电压降模型
$$
\Delta V(t) = A_f(1 - e^{-t/\tau_f}) + A_m(1 - e^{-t/\tau_m}) + A_t(1 - e^{-t/\tau_t})
$$
