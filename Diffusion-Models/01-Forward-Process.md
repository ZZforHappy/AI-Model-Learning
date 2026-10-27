# 01 正向扩散（Forward Diffusion Process）

## 1. 目标与基本设定

假设真实数据为 \(x_0\in\mathbb R^d\)，其分布 \(q(x_0)\) 通常比较复杂。正向扩散通过逐步添加高斯噪声，使 \(x_T\) 的分布接近标准高斯分布。正向过程是人为指定的，不需要神经网络训练。

选择噪声方差序列（noise schedule）

\[
0<\beta_t<1,\qquad t=1,\ldots,T,
\]

并定义

\[
\alpha_t=1-\beta_t,\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
=\alpha_1\alpha_2\cdots\alpha_t,\qquad
\bar{\alpha}_0=1.
\]

这里 \(\prod\) 表示**连乘**，下标 \(s=1\) 是起点，上标 \(t\) 是终点；\(\bar{\alpha}_t\) 是前 \(t\) 步所有 \(\alpha_s\) 的乘积，而不是某一步的 \(\alpha_t\)。

## 2. 单步正向扩散

令各时间步的噪声 \(\epsilon_t\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,I)\)，且与 \(x_0\) 独立。定义

\[
x_t=\sqrt{\alpha_t}\,x_{t-1}
+\sqrt{1-\alpha_t}\,\epsilon_t.
\tag{1}
\]

给定 \(x_{t-1}\) 时，第一项是确定的，第二项是均值为零、协方差为 \((1-\alpha_t)I\) 的高斯向量，因此

\[
\mathbb E[x_t\mid x_{t-1}]=\sqrt{\alpha_t}\,x_{t-1},
\]

\[
\operatorname{Cov}(x_t\mid x_{t-1})=(1-\alpha_t)I.
\]

所以单步条件分布为

\[
\boxed{
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{\alpha_t}\,x_{t-1},
(1-\alpha_t)I
\right).
}
\tag{2}
\]

符号 \(\mathcal N(x;\mu,\Sigma)\) 表示在 \(x\) 处求值、均值为 \(\mu\)、协方差为 \(\Sigma\) 的高斯密度。

由于每一步只依赖上一状态与新噪声，整个过程是马尔可夫链：

\[
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
\]

平方根的作用是控制**方差**权重：独立噪声项的协方差等于系数平方乘以 \(I\)。不能因此断言任意原始数据的协方差始终为 \(I\)。

## 3. 从第一步推导到第二步

第一步：

\[
x_1=\sqrt{\alpha_1}\,x_0
+\sqrt{1-\alpha_1}\,\epsilon_1.
\tag{3}
\]

第二步：

\[
x_2=\sqrt{\alpha_2}\,x_1
+\sqrt{1-\alpha_2}\,\epsilon_2.
\]

代入式 (3)，逐项展开：

\[
\begin{aligned}
x_2
&=\sqrt{\alpha_2}
\left(\sqrt{\alpha_1}\,x_0
+\sqrt{1-\alpha_1}\,\epsilon_1\right)
+\sqrt{1-\alpha_2}\,\epsilon_2\\
&=\sqrt{\alpha_1\alpha_2}\,x_0
+\underbrace{
\sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1
+\sqrt{1-\alpha_2}\,\epsilon_2
}_{z_2}.
\end{aligned}
\tag{4}
\]

现在研究噪声和 \(z_2\)。因为 \(\epsilon_1,\epsilon_2\) 相互独立且均为标准高斯向量，它们的线性组合仍是高斯向量，并且

\[
\mathbb E[z_2]=0.
\]

利用独立性，交叉协方差为零：

\[
\begin{aligned}
\operatorname{Cov}(z_2)
&=\alpha_2(1-\alpha_1)I+(1-\alpha_2)I\\
&=\left(\alpha_2-\alpha_1\alpha_2+1-\alpha_2\right)I\\
&=(1-\alpha_1\alpha_2)I.
\end{aligned}
\tag{5}
\]

因此

\[
z_2\sim\mathcal N(0,(1-\alpha_1\alpha_2)I).
\]

一个标准高斯向量 \(\epsilon\sim\mathcal N(0,I)\) 乘以 \(\sqrt{1-\alpha_1\alpha_2}\) 后，恰好具有相同分布。所以给定 \(x_0\)，

\[
\boxed{
x_2\overset{d}{=}
\sqrt{\alpha_1\alpha_2}\,x_0
+\sqrt{1-\alpha_1\alpha_2}\,\epsilon.
}
\tag{6}
\]

\(\overset d=\) 表示**同分布**，不是说不同噪声样本逐点相等。

## 4. 从第二步推广到任意第 t 步

用数学归纳法。假设在第 \(t-1\) 步已经成立：

\[
x_{t-1}\mid x_0
\overset d=
\sqrt{\bar{\alpha}_{t-1}}\,x_0
+\sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon',
\]

其中 \(\epsilon'\sim\mathcal N(0,I)\)，并且它与新加入的 \(\epsilon_t\) 独立。

将其代入式 (1)：

\[
\begin{aligned}
x_t\mid x_0
\overset d={}&
\sqrt{\alpha_t\bar{\alpha}_{t-1}}\,x_0\\
&+\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\,\epsilon'
+\sqrt{1-\alpha_t}\,\epsilon_t.
\end{aligned}
\tag{7}
\]

后两项的协方差为

\[
\begin{aligned}
&\left[\alpha_t(1-\bar{\alpha}_{t-1})+(1-\alpha_t)\right]I\\
&=\left(1-\alpha_t\bar{\alpha}_{t-1}\right)I\\
&=(1-\bar{\alpha}_t)I,
\end{aligned}
\tag{8}
\]

其中最后一步用到了连乘关系

\[
\bar{\alpha}_t
=\prod_{s=1}^{t}\alpha_s
=\alpha_t\prod_{s=1}^{t-1}\alpha_s
=\alpha_t\bar{\alpha}_{t-1}.
\tag{9}
\]

因此两项独立高斯噪声可以合并为一个同分布的高斯噪声，得到核心闭式表达式：

\[
\boxed{
x_t\mid x_0\overset d=
\sqrt{\bar{\alpha}_t}\,x_0
+\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
}
\tag{10}
\]

其对应的条件分布为

\[
\boxed{
q(x_t\mid x_0)=
\mathcal N\!\left(
x_t;\sqrt{\bar{\alpha}_t}\,x_0,
(1-\bar{\alpha}_t)I
\right).
}
\tag{11}
\]

这说明：**给定原始数据 \(x_0\)，可以直接采样任意时刻的 \(x_t\)，不必真的执行前 \(t\) 次加噪。**

## 5. 为什么最终接近标准高斯分布

若噪声日程选择得使

\[
\bar{\alpha}_T=\prod_{s=1}^{T}(1-\beta_s)\approx0,
\]

则式 (10) 中的原始信号系数接近零，而噪声系数接近一：

\[
x_T\mid x_0\approx\epsilon,
\qquad
q(x_T\mid x_0)\approx\mathcal N(0,I).
\]

这是**分布上的近似**，有限步时一般不是严格相等。

## 6. 本章结论

正向扩散最重要的两个公式：

**单步转移**

\[
q(x_t\mid x_{t-1})=
\mathcal N\!\left(
x_t;\sqrt{\alpha_t}\,x_{t-1},(1-\alpha_t)I
\right).
\]

**任意时刻的闭式分布**

\[
q(x_t\mid x_0)=
\mathcal N\!\left(
x_t;\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I
\right),
\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s.
\]

下一章：[02 反向扩散](02-Reverse-Process.md)。
