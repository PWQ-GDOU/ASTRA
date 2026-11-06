# 控制方程

## 一、电磁场方程（实体模型）

### 矢量磁位方程
$$
\nabla \times \left(\frac{1}{\mu_0 \mu_r} \nabla \times \mathbf{A}\right) + \sigma \frac{\partial \mathbf{A}}{\partial t} = \mathbf{J}_e
$$

### 永磁体建模
剩余磁通密度 (退磁故障注入):
$$
B_r(\gamma) = B_{r0} \cdot (1 - 0.4\gamma) = 1.2 \, \text{T} \cdot (1 - 0.4\gamma)
$$

## 二、BLDC 电机方程（系统级 ODE 模型）

### 电气子模型

A 相电压方程:
$$
V_a = R_{phase} \cdot i_a + L_{phase} \frac{di_a}{dt} + E_a
$$

B 相电压方程:
$$
V_b = R_{phase} \cdot i_b + L_{phase} \frac{di_b}{dt} + E_b
$$

C 相电压方程:
$$
V_c = R_{phase} \cdot i_c + L_{phase} \frac{di_c}{dt} + E_c
$$

其中相电压 (PWM 等效):
$$
V_x = \frac{1}{2} V_{dc} \cdot \text{duty} \cdot f_x(\theta_e), \quad x \in \{a,b,c\}
$$

反电势 (考虑退磁):
$$
E_x = K_{e0} \cdot B_{r,scale} \cdot \omega \cdot f_x(\theta_e)
$$

$$
B_{r,scale} = \max(0.05, 1 - dBr)
$$

匝间短路影响:
$$
R_{phase} = R_0 \cdot \max(0.05, 1 - dR) \cdot \max(0.05, 1 + \alpha_{Cu}(T_{wind} - T_{amb}))
$$

$$
L_{phase} = L_0 \cdot \max(0.10, 1 - 0.5 \cdot dR)
$$

### 机械子模型
$$
J_{wheel} \frac{d\omega}{dt} = T_{em} - T_{fric} - T_{load}
$$

电磁转矩:
$$
T_{em} = K_{t0} \cdot B_{r,scale} \cdot \sum_{x=a,b,c} f_x(\theta_e) \cdot i_x
$$

非线性摩擦 (含退化):
$$
T_{fric} = \text{sign}(\omega) \cdot \left[T_0 + B_0|\omega| + T_{cog} \cdot \cos(n_{cog} \cdot \theta)\right] + T_{wear}
$$

$$
T_{wear} = 6 \times 10^{-3} \cdot \gamma^{1.7} \quad [\text{N·m}]
$$

### 热子模型
$$
C_{th} \frac{dT_{wind}}{dt} = P_{cu} + \eta_{fric} \cdot |T_{fric} \cdot \omega| - \frac{T_{wind} - T_{amb}}{R_{th}}
$$

铜耗:
$$
P_{cu} = R_{phase} \cdot (i_a^2 + i_b^2 + i_c^2)
$$

### 振动遥测
$$
a_{vib} = vib\_gain \cdot (\omega^2 \cdot \gamma + |T_{wear}|) \quad [\text{m/s}^2]
$$

## 三、三相反电势归一化函数

$$
f_a(\theta_e) = \sin(\theta_e)
$$
$$
f_b(\theta_e) = \sin(\theta_e - 2\pi/3)
$$
$$
f_c(\theta_e) = \sin(\theta_e + 2\pi/3)
$$

其中电角度: $\theta_e = p \cdot \theta$ (p=4 为极对数)

## 四、速度环控制
$$
\text{duty} = \min\left(1, \max\left(0, 0.40 + K_{p,speed} \cdot (\omega_{cmd} - \omega)\right)\right)
$$
