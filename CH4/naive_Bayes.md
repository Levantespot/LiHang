# naive Bayes

## 特征

* 通过先验概率求出后验概率最大的输出 $y$
* 基于贝叶斯定理和特征条件独立假设
* 生成模型

## 前置知识

1.条件独立假设（independent and identically distributed）(*i.i.d.* or *iid* or **IID**)

条件独立假设：给定 $Y$ 的情况下求 $P(Y|X)$，其中$X\in\{X_1,X_2,\dots,X_n\}$：

1. 每一个$X_i$和其他的每个$X_k$是条件独立的
1. 每一个$X_i$和其他的每个$X_k$的子集是条件独立的

$$
\begin{align}
P(X=x|Y=c_k)&=P(X^{(1)},\dots,X^{(n)}|Y=c_k)\\
&=\prod^n_{j=1}P(X^{(j)}=x^{(j)}|Y=c_k)
\end{align}
$$

## 模型

输入：$x_i \in \cal{X} \subseteq{\bf{R}^n}$ 为实例的特征向量；

输出：$y_i \in \cal{Y} = {c_1, c_2,\cdots,c_K}$ 为实例的类别，$i=1,2,\cdots,N$。

决策：

1. 学习先概率分布 $P(Y=c_k),\quad k=1,2,\cdots,K$

2. 学习条件概率分布
   $$
   P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},X^{(2)}=x^{(2)},\cdots,X^{(n)}=x^{(n)}|Y=c_k),\quad k=1,2,\cdots,K
   $$
   假设 $x^{(j)}$ 可取值有 $S_j$ 个，$j=1,2,\cdots,n$，则：
   $$
   P(X^{(j)}=x^{(j)})=P(x^{(j,k)})=1-\sum_{i=1,i\neq k}^{S_j}P(x^{(j,i)})
   $$
   该式共 $S_j$ 个参数，即 $P(x^{(j,1)}),P(x^{(j,2)}),\cdots,P(x^{(j,S_j)})$。假设 $Y$ 可取值有 $K$ 个，则条件概率分布的参数个数总数为 $K\prod_{j=1}^{n}{S_j}$。

   再根据条件独立性的假设（用于分类的特征在类确定的条件下是条件独立的，简化算法，损失准确度），得
   $$
   P(X=x|Y=c_k)=\prod_{j=1}^{n}{P(X^{(j)}|Y=c_k)}
   $$

3. 计算后验概率 $P(Y=c_k|X=x)$，将后验概率最大的类作为 $x$ 的类输出。
   $$
   \begin{aligned}
   y=f(x)&=\arg \max_{c_k} P(Y=c_k|X=x) \\
   &=\arg \max_{c_k} \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_k{P(X=x|Y=c_k)P(Y=c_k)}} \\
   &=\arg \max_{c_k} \frac{P(Y=c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_k{P(Y=c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}}
   \end{aligned}
   $$
   由于对任意的 $c_k$ 分母相等，故上式可等价于：
   $$
   y=f(x)=\arg \max_{c_k} {P(Y=c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}
   $$

## 策略

假设采用 0-1 损失函数：
$$
L（Y,f(x))=\begin{cases}
1,&Y\neq f(X)\\
0,&Y=f(X)
\end{cases}
$$
则期望风险函数为：
$$
\begin{aligned}
R_{\exp}(f)&=E[L(Y,f(X))]\\
&=E_X\sum_{k=1}^{K}[L(c_k,f(X))]P(c_k|X)
\end{aligned}
$$
若要使期望风险函数最小化，只需对每个 $X=x$ 逐个极小化，有：
$$
\begin{aligned}
\arg \min R_{\exp}(f)&\Leftrightarrow \arg \min_{y\in\cal{Y}}\sum_{k=1}^{K}{L(c_k,y)P(c_k|X=x)}\\
&=\arg \min_{y\in\cal{Y}}\sum_{k=1}^{K}{P(c_k\neq y|X=x)}\\
&=\arg \min_{y\in\cal{Y}}(1-\sum_{k=1}^{K}{P(c_k=y|X=x)})\\
&\Leftrightarrow\arg \max_{y\in\cal{Y}}{P(c_k=y|X=x)}\\
\end{aligned}
$$
即 $\text{风险最小化}\Rightarrow\text{后验概率最大化}$ ，若想要使风险最小化，就需要后验概率最大化。

## 算法

### 极大似然估计

输入：训练集 $T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N),\}$，其中 $x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T$，$x_i^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特诊，$x_i^{(j)}\in\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$，$s_{jl}$ 是第 $j$ 个特征可能的第 $l$ 个取值，$j=1,2,\cdots,n$，$l=1,2,\cdots,S_j$，$y_i\in\{c_1,c_2,\cdots,c_K\}$；

输出：实例 $x$ 的分类。

算法：

1. 计算先验概率及条件概率
   $$
   P(Y=c_k)=\frac{\sum_{i=1}^{N}{I(y_i=c_k)}}{N}\\
   P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N}{I(x_i^{(j)}=a_{jl},y_i=c_k)}}{\sum_{i=i}^{N}{I(y_i=c_k)}}\\
   j=1,2,\cdots,n;\quad l=1,2,\cdots,S_j;\quad k=1,2,\cdots,K;
   $$

2. 对于给定的实例 $x=(x^{(1)},x^{(2)},\cdots,x^{(n)})^T$ 其类为：
   $$
   y=\arg \max_{c_k}P(Y=c_k)\prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_k),\quad k=1,2,\cdots,K
   $$
   
3. 

### 贝叶斯估计

用极大似然估计可能出现某一条件概率（$j=m$）为 0 导致其余的条件概率（$j\neq m$）不起作用，导致后验概率为 0，影响分类效果。解决方法是采用贝叶斯估计：
$$
P_{\lambda}(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^{N}{I(x_i^{(j)}=a_{jl},y_i=c_k)}+\lambda}{\sum_{i=i}^{N}{I(y_i=c_k)}+S_j\lambda}
$$
常取 $\lambda=1$，称为拉普拉斯平滑（Laplacian smoothing）。