# Gradient Descent

梯度下降法（Gradient Descent）也叫最速下降法（steepest descent）是求解**无约束**最优化问题的一种最常用的方法。

无约束优化问题通常形式为：
$$
\min_{x\in\R^n}f(x)
$$
其中 $f(x)$ 是 $\R^n$ 上具有一阶连续偏导数的函数，$x^*$ 表示目标函数 $f(x)$ 的极小点。

梯度下降法是一种迭代算法，通过选取适当的初值 $x^{(0)}$，沿着函数数值下降的最快的方向（即负梯度方向）不断迭代更新 $x$ 的值，进行目标函数的极小化，第 $k$ 次迭代值为 $x^{(k)}$，直到目标函数收敛。算法如下：

**梯度下降法**

输入：目标函数 $f(x)$，梯度函数 $g(x)=\nabla f(x)$，计算精度 $\epsilon$；

输出：$f(x)$ 的极小点 $x^*$。

算法：

1. 取初值 $x^{(0)}\in \R^n$，置 $k=0$；

2. 计算 $f(x^{(k)})$；

3. 计算梯度 $g_k=g(x^{(k)})$，当 $||g_k||<\epsilon$ 时，停止迭代，令 $x^*=x(k)$；否则，令 $p_k=-g(x^{(k)})$，求 $\lambda_k$，使 
   $$
   f(x^{(k)}+\lambda_k p_k)=\min_{\lambda\geq0}{f(x^{(k)}+\lambda p_k)}
   $$

4. 置 $x^{(k+1)}=x^{(k)}+\lambda_k p_k$，计算 $f(x^{(k+1)})$ ；当 $||f(x^{(k+1)}-f(x^{(k)}))||<\epsilon$ 或 $||x^{(k+1)-x^{(k)}}<\epsilon$ 时，停止迭代，令 $x^*=x^{(k+1)}$。

5. 否则，置 $k=k+1$，转 $3$。

当目标函数是凸函数时，梯度下降的解释全局最优解。一般情况，不保证是全局最优。收敛速度不定。