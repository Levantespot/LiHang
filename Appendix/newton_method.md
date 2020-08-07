# Newton method

牛顿法（Newton method）和拟牛顿法（quasi-Newton method）是求解无约束最优化问题的常用方法，收敛速度快。

## 前置知识

### 黑塞矩阵（Hessian matrix）

**二元函数的黑塞矩阵**

一元函数 $f(x)$ 在 $x^{(0)}$ 处的泰勒展开式：
$$
f(x)=f(x^{(0)})+f^\prime(x^{(0)})\Delta{x}+f^{\prime\prime}(x^{(0)}){\Delta{x}^2}+\cdots
$$
其中 $\Delta x=x-x^{(0)},\ \Delta{x}^2=(x-x^{(0)})^2$。

二元函数 $f(x_1,x_2)$ 在 $X^{(0)}=(x_1^{(0)},x_2^{(0)})$ 处的泰勒展开式为：
$$
\begin{align}
f(x_1,x_2)
&=f(x_1^{(0)},x_2^{(0)}) + \frac{\partial{f}}{\partial{x_1}}\bigg|_{X^{(0)}}\Delta{x_1} + \frac{\partial{f}}{\partial{x_2}}\bigg|_{X^{(0)}}\Delta{x_2} + \\
&\frac{1}{2}{\left[ \frac{\partial^2{f}}{\partial{4x_1^2}}\bigg|_{X^{(0)}}\Delta{x_1}^2 +
\frac{\partial^2{f}}{\partial{x_2^2}}\bigg|_{X^{(0)}}\Delta{x_2}^2 +
2\frac{\partial^2{f}}{\partial{x_1}\partial{x_2}}\bigg|_{X^{(0)}}\Delta{x_1}\Delta{x_2}
\right]} + \cdots
\end{align}
$$
其中 $\Delta x=x-x^{(0)},\ \Delta{x}^2=(x-x^{(0)})^2$。

将上述展开式改写成矩阵形式：
$$
f(X) = f(X^{(0)}) + \left( \frac{\partial{f}}{\partial{x_1}},\frac{\partial{f}}{\partial{x_2}} \right)_{X^{(0)}}{\begin{pmatrix}\Delta{x_1} \\ \Delta{x_2}\end{pmatrix}} +
\frac{1}{2}(\Delta x_1,\Delta x_2){\begin{pmatrix} \frac{\partial^2{f}}{\partial{x_1}^2} &\frac{\partial^2{f}}{\partial{x_1}\partial{x_2}}\\ \frac{\partial^2{f}}{\partial{x_2}\partial{x_1}} &\frac{\partial^2{f}}{\partial{x_2}^2} \end{pmatrix}}\Bigg|_{X^{(0)}} {\begin{pmatrix} \Delta x_1 \\  \Delta{x_2} \end{pmatrix}} + \cdots
$$
即：
$$
f(X)=f(X^{(0)})+\nabla{f(X^{(0)})}^T\Delta{X} + \frac{1}{2}\Delta{X^T}G(X^{(0)})\Delta{X} + \cdots
$$
其中：
$$
G(X^{(0)})=
\begin{pmatrix}
\frac{\partial^2{f}}{\partial{x_1}^2} &\frac{\partial^2{f}}{\partial{x_1}\partial{x_2}}\\ \frac{\partial^2{f}}{\partial{x_2}\partial{x_1}} &\frac{\partial^2{f}}{\partial{x_2}^2}
\end{pmatrix}\Bigg|_{X^{(0)}};\quad\Delta{X}={\begin{pmatrix} \Delta x_1 \\  \Delta{x_2} \end{pmatrix}}
$$
$G(X^{(0)})$ 是 $f(x_1,x_2)$ 在 $X^{(0)}$ 处的二阶偏导数所组成的方阵。

**多元函数的黑塞矩阵**

将二元函数的泰勒展开式推广到多元函数，则 $f(x_1,x_2,\cdots,x_n)$ 在 $X^{(0)}$ 点处的泰勒展开式的矩阵形式为：
$$
f(X)=f(X^{(0)})+\nabla f(X^{(0)})^T \Delta X+\frac{1}{2}\Delta{X^T}G(X^{(0)})\Delta{X}+\cdots
$$
其中：
$$
\Delta f(X^{(0)})=\left[ \frac{\partial{f}}{\partial{x_1}},\frac{\partial{f}}{\partial{x_2}},\cdots,\frac{\partial{f}}{\partial{x_n}} \right]\bigg|_{X^{(0)}}^T;
\quad
G(X^{(0)})=\begin{bmatrix}
\frac{\partial^2{f}}{\partial{x_1}^2} &\frac{\partial^2{f}}{\partial{x_1}\partial{x_2}} &\cdots &\frac{\partial^2{f}}{\partial{x_1}\partial{x_n}}\\
\frac{\partial^2{f}}{\partial{x_2}\partial{x_1}} &\frac{\partial^2{f}}{\partial{x_2}^2} &\cdots &\frac{\partial^2{f}}{\partial{x_2}\partial{x_n}}\\
\vdots &\vdots &\ddots &\vdots\\
\frac{\partial^2{f}}{\partial{x_n}\partial{x_1}} &\frac{\partial^2{f}}{\partial{x_n}\partial{x_2}} &\cdots &\frac{\partial^2{f}}{\partial{x_n}\partial{x_n}}\\
\end{bmatrix}
$$
$G(X^{(0)})$ 是 $f(X)$ 在 $X^{(0)}$ 处的二阶偏导数所组成的 $n\times n$ 阶对称矩阵。

**特性**

1. 当 $G(X^{(0)})$ 为正定矩阵（positive definite matrix）时， 在 $X$ 处为极小值；
2. 当 $G(X^{(0)})$ 为负定矩阵（negative definite matrix）时， 在 $X$ 处是极大值；
3. 当 $G(X^{(0)})$ 为不定矩阵时， $X$ 不是极值点。
4. 当 $G(X^{(0)})$ 为半正定矩阵或半负定矩阵时，$X$ 是“可疑”极值点，尚需要利用其他方法来判定。

## 算法

### 牛顿法

**思路**

考虑无约束最优化问题：
$$
\min_{x\in \R^n}f(x)
$$
其中 $x^*$ 为目标函数的极小点。

假设 $f(x)$ 具有二阶连续偏导数，若第 $k$ 次迭代值为 $x^{(k)}$，则 $f(x)$ 在 $x^{(k)}$ 点处的二阶泰勒展开式为：
$$
f(x)=f(x^{(k)})+g_k^T(x-x^{(k)})+\frac{1}{2}(x-x^{(k)})^TH(x^{(k)})(x-x^{(k)})
$$
其中，$g_k=g(x^{(k)})=\nabla f(x^{(k)})$ 是 $f(x)$ 的梯度向量在点 $x^{(k)}$ 的值， $H(x^{(k)})$ 是 $f(x)$ 的黑塞矩阵：
$$
H(x)=\left[ \frac{\partial^2f}{\partial x_i \partial x_j} \right]_{n\times n}
$$
若函数 $f(x)$ 在 $x$ 处有极值，则必要条件为一阶导数为 0，即 $\nabla f(x)=0$；特别地，当 $H(X^{(k)})$ 为正定矩阵时，函数 $f(x)$ 的极值为极小值。

对 $f(x)$ 的左右两边求导可得：
$$
\nabla f(x)=g_k+H_k(x-x^{(k)})
$$
其中 $H_k=H(x^{(k)})$。令下一次迭代值 $x^{(k+1)}$ 为极值，有 $\nabla f(x^{(k+1)})=0$，代入上式得：
$$
\nabla f(x^{(k+1)})=g_{k+1}=0=g_k+H_k(x^{(k+1)}-x^{(k)})\\
x^{(k+1)}=x^{(k)}-H_k^{-1}g_k
$$
记 $p_k=-H_k^{-1}g_k$，则 $H_kp_k=-g_k$，故有：
$$
x^{(k+1)}=x^{(k)}+p_k
$$
**算法**

输入：目标函数 $f(x)$，梯度 $g(x)=\nabla f(x)$，黑塞矩阵 $H(x)$，精度要求 $\epsilon$；

输出：$f(x)$ 的极小值点 $x^*$。

算法：

1. 取初始点 $x^{(0)}$，置 $k=0$；
2. 计算 $g_k=g(x^{(k)})$；
3. 若 $||g_k||<\epsilon$，则停止计算，得到近似解 $x^*=x^{(k)}$；
4. 否则计算 $H_k=H(x^{(k)})$，并求 $p_k=-H_k^{-1}g_k$；
5. 置 $x^{(k+1)}=x^{(k)}+p_k$；
6. 置 $k=k+1$，转 $2$，直到满足 $3$。

由于 $4$ 中求黑塞矩阵的逆矩阵计算比较复杂，且可能不可逆，故有其他改进方法。

### 拟牛顿法

**思路**

泰勒展开式中，只保留一阶梯度，并考虑用一个 $n$ 阶矩阵 $G_k=G(x^{(k)})$ 来近似代替 $H_K^{-1}=H^{-1}(x^{(k)})$ 或 用 $B_k$ 来近似代替 $H_k$。要找到近似的替代矩阵，必定要和 $H_k$ 有类似的性质。近似矩阵 $G_k$ 需满足一些条件：

对 $f(x)$ 在 $x^{k+1}$ 处的泰勒展开式左右两边求导，并带入 $x^{(k)}$ 即 $x=x^{(k)}$ 得：
$$
g_{k+1}-g_k=H_{k+1}(x-x^{(k)})
$$
记 $g_{k+1}-g_k=y_k,\ x^{(k+1)}-x^{(k)}=\delta_k$，得：
$$
y_k=H_{k+1}\delta_k \Longleftrightarrow H_{k+1}^{-1}y_k=\delta_k
$$
由于 $H_{k+1}$ 正定，则 $H_{k+1}^{-1}$ 正定，故 $G_k$ 应满足**拟牛顿条件**如下：
$$
G_{k+1}y_k=\delta_k
$$
由于：
$$
x=x^{(k)}+p_k=x^{(k)}-H_k^{-1}g_k
$$
则有：
$$
x=x^{(k)}+\lambda p_k=x^{(k)}-\lambda H_k^{-1}g_k
$$
其中 $\lambda>0$，带入 $f(x)$ 的泰勒展开式，并舍弃二阶及以后的部分，得：
$$
f(x)=f(x^{(k)})-\lambda g_k^T H_k^{-1} g_k
$$
由于 $H_k$ 正定，则 $H_k^{-1}$ 正定，故有 $g_k^T H_k^{-1} g_k > 0$。

#### DFP（Davidon-Fletcher-Powell）算法（DFP algorithm）

**思路**

假设每一步迭代中矩阵 $G_{k+1}$ 是由 $G_k$ 加上两个附加项 $P_k, Q_k$ 构成的，即：
$$
G_{k+1}=G_k+P_k+Q_k\\
G_{k+1}y_k=G_k y_k + P_k y_k + Q_k y_k
$$
其中有 $P_k y_k=\delta_k,\ Q_k y_k=-G_k y_k$，可取：
$$
P_k=\delta_k \delta_k^T(\delta_k^T)^{-1}(y_k)^{-1}=\frac{\delta_k \delta_k^T}{\delta_k^T y_k}\\
Q_k=-G_k y_k y_k^T G_k(y_k^T G_k)^{-1}(y_k)^{-1}=-\frac{G_k y_k y_k^T G_k}{y_k^T G_k y_k}
$$
则 $G_{k+1}$ 的迭代公式为：
$$
G_{k+1}=G_k+\frac{\delta_k \delta_k^T}{\delta_k^T y_k}-\frac{G_k y_k y_k^T G_k}{y_k^T G_k y_k}
$$
**算法**

输入：目标函数 $f(x)$，梯度 $g(x)=\nabla f(x)$，精度要求 $\epsilon$；

输出： $f(x)$ 的极小值点 $x^*$；

算法：

1. 选定初始点 $x^{(0)}$，取 $G_0$ 为正定对称矩阵，置 $k=0$；
2. 计算 $g_k=g(x^{(k)})$ 不写了，看书

#### BFGS（Broyden-Fletcher-Goldfarb-Shanno）算法（BFGS algorithm）

**思路**

用 $B_k$ 逼近 $H_k$，此时相应的牛顿条件是：
$$
B_{k+1}\delta_k=y_k
$$
同样令：
$$
B_{k+1}=B_k+P_k+Q_k\\
B_{k+1}\delta_k=B_k\delta_k+P_k\delta_k+Q_k\delta_k
$$
考虑使 $P_k$ 和 $Q_k$ 满足：
$$
P_k\delta_k=y_k\\
Q_k\delta_k=-B_k\delta_k
$$
则有：
$$
B_{k+1}=B_k+\frac{y_k y_k^T}{y_k^T \delta_k}-\frac{B_k \delta_k \delta_k^T B_k}{\delta_k^T B_k \delta_k}
$$
**算法**

输入：目标函数 $f(x)$，梯度 $g(x)=\nabla f(x)$，精度要求 $\epsilon$；

输出： $f(x)$ 的极小值点 $x^*$；

算法：

1. 选定初始点 $x^{(0)}$，取 $B_0$ 为正定对称矩阵，置 $k=0$；

2. 计算 $g_k=g(x^{(k)})$。若 $||g_k||<\epsilon$，则停止计算，得近似解 $x^*=x^{(k)}$；

3. 由 $B_k p_k=-g_k$ 求出 $p_k$；

4. 一维搜索：求 $\lambda_k$ 使得：
   $$
   f(x^{(k)}+\lambda_k p_k)=\min_{\lambda \geq0}f(x^{(k)}+\lambda p_k)
   $$

5. 置 $x^{(k+1)}=x^{(k)}+\lambda_k p_k$；

6. 计算 $g_{k+1}=g(x^{(k+1)})$，若 $||g_{k+1}||<\epsilon$，则停止计算，得近似解 $x^*=x^{(k+1)}$；

7. 否则根据 $B_{k+1}=B_k+\frac{y_k y_k^T}{y_k^T \delta_k}-\frac{B_k \delta_k \delta_k^T B_k}{\delta_k^T B_k \delta_k}$计算 $B_{K+1}$；

8. 置 $k=k+1$，转 $3$。

#### Broyden 类算法（Broden’s algorithm）

pass

