# Lagrange duality

### 原始问题

假设 $f(x),\ c_i(x), h_j(x)$ 是定义在 $\R^n$ 上的连续可微函数。考虑最优化问题：

$$
\begin{align}
\min_{x\in \R^n}\quad &f(x) \\
\mathrm{s.t.}\quad &c_i(x)\leq 0,\quad i=1,2,\cdots,k \\
&h_j(x)=0, \quad j=1,2,\cdots,l
\end{align}
$$
称此约束最优化问题为原始最优化额问题或原始问题。

### 广义拉格朗日函数（generalized Lagrange function）

广义拉格朗日函数形式为：
$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^k{\alpha_i c_i (x)}+\sum_{j=1}^l{\beta_i j_j (x)}
$$
其中，$x=(x^{(1)},x^{(2)},\cdots,x^{(n)})^T\in\R^n$，$\alpha_i,\ \beta_j$ 是拉格朗日乘子，$\alpha_i\geq0$。考虑 $x$ 的函数：
$$
\theta_P(x)=\max_{\alpha,\beta;\ \alpha_i\geq0}L(x,\alpha,\beta)
$$
其中，下标 P​ 表示原始问题，易知：
$$
\theta_P(x)=
\begin{cases}
& f(x),\quad x\ 满足原始问题约束\\
& +\infin, \quad 其他
\end{cases}
$$
考虑极小化问题：
$$
\min_{x}\theta_P(x)=\min_x\max_{\alpha,\beta;\ \alpha_i\geq0}L(x,\alpha,\beta)
$$
易知该极小化问题**等价于**原始问题。定义原始问题的最优值：
$$
p^*=\min_{x}\theta_P(x)
$$
为原始问题的值。

### 对偶问题

定义：
$$
\theta_D(x)=\min_{\alpha,\beta;\ \alpha_i\geq0}L(x,\alpha,\beta)
$$
再考虑极大化 $\theta_D(\alpha,\beta)$ 问题得：
$$
\max_{\alpha,\beta;\ \alpha_i\geq0}\theta_D(\alpha,\beta)=\max_{\alpha,\beta;\ \alpha_i\geq0}\min_xL(x,\alpha,\beta)
$$
问题 $\max\limits_{\alpha,\beta;\ \alpha_i\geq0}\min_\limits{x}L(x,\alpha,\beta)$ 称为广义拉格朗日函数的极大极小问题，即：
$$
\max_{\alpha,\beta}\theta_D(\alpha,\beta)=\max_{\alpha,\beta}\min_xL(x,\alpha,\beta)\\
\mathrm{s.t.}\quad \alpha_i\geq0,\quad i=1,2,\cdots,k
$$
称为原始问题的对偶问题。定义对偶问题的最优值：
$$
d^*=\max_{\alpha,\beta;\ \alpha_i\geq0}\theta_D(\alpha,\beta)
$$
为对偶问题的值。

### 原始问题与对偶问题的关系。

**定理1** 若原始问题与对偶问题都有最优值，则：
$$
d^*=\max_{\alpha,\beta}\min_xL(x,\alpha,\beta) \leq \min_x\max_{\alpha,\beta;\ \alpha_i\geq0}L(x,\alpha,\beta)=p^*
$$
**推论1** 设 $x^*,\alpha^*,\beta^*$ 分别是原始问题和对偶问题的可行解，且 $d^*=p^*$，则该组解分别是原始问题和对偶问题的最优解。

以下定理看不懂：

**定理2** 考虑原始问题和对偶问题，假设 $f(x)$ 和 $c_i(x)$ 是凸函数，$h_j(x)$ 是仿射函数（最高次数为 1），且不等式约束 $c_i(x)$ 严格可行，及存在 $x$ 对所有 $i$ 有 $c_i(x)<0$，则存在 $x^*,\alpha^*,\beta^*$，使 $x^*$ 是原始问题的解，$\alpha^*$，$\beta^*$是对偶问题的解，且：
$$
p^*=d^*=L(x^*,\alpha^*,\beta^*)
$$
**定理3** 满足`定理2` 的条件的情况下，$x^*,\alpha^*,\beta^*$ 分别是原始问题和对偶问题的解的充分必要条件是 $x^*,\alpha^*,\beta^*$ 满足下面的 KKT 条件：
$$
\nabla_xL(x^*,\alpha^*,\beta^*)=0\\
\alpha_i^*c_i(x^*)=0,\quad i=1,2,\cdots,k\\
c_i(x^*)\leq0,\quad i=1,2,\cdots,k\\
\alpha_i^*\geq0,\quad i=1,2,\cdots,k\\
h_j(x^*)=0,\quad j=1,2,\cdots,l
$$
其中第二项称为 KKT 的对偶互补条件，得当 $a_i^*>0$，则 $c_i(x^*)=0$。