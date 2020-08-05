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

1. 当 $G(X^{(0)})$ 为正定矩阵时， 在 $X$ 处为极小值；
2. 当 $G(X^{(0)})$ 为负定矩阵时， 在 $X$ 处是极大值；
3. 当 $G(X^{(0)})$ 为不定矩阵时， $X$ 不是极值点。
4. 当 $G(X^{(0)})$ 为半正定矩阵或半负定矩阵时，$X$ 是“可疑”极值点，尚需要利用其他方法来判定。

## 算法

### 1.牛顿法

考虑无约束最优化问题：
$$
\min_{x\in \R^n}f(x)
$$
其中 $x^*$ 为目标函数的极小点。

