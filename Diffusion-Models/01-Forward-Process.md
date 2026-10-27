# 01 正向扩散（Forward Process）

## 目标
理解如何将复杂数据分布逐渐转化为近似标准高斯分布，以及为什么可以一步采样任意时间的含噪数据。

## 单步加噪
令 $\alpha_t=1-\beta_t$，其中 $0<\beta_t<1$，各步噪声独立且 $\epsilon_t\sim\mathcal N(0,I)$。

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

因此

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I).
$$

这是一条马尔可夫链：给定 $x_{t-1}$，下一状态与更早状态条件独立。

## 两步推导
将第一步代入第二步：

$$
x_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2.
$$

后两项是独立高斯噪声的线性组合，均值为零，协方差为

$$
[\alpha_2(1-\alpha_1)+(1-\alpha_2)]I=(1-\alpha_1\alpha_2)I.
$$

因此，在给定 $x_0$ 时，$x_2$ 与下式同分布：

$$
x_2\overset{d}{=}\sqrt{\alpha_1\alpha_2}x_0+\sqrt{1-\alpha_1\alpha_2}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

## 推广到任意 t
定义 $\bar\alpha_t=\prod_{s=1}^t\alpha_s$，递推得到：

$$
\boxed{x_t\overset{d}{=}\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon}
$$

$$
\boxed{q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)}
$$

这说明训练时可以随机选择 $t$，直接生成 $x_t$，无需逐步加噪。

若 $\bar\alpha_T\approx0$，则 $q(x_T\mid x_0)\approx\mathcal N(0,I)$。注意这是近似的分布结论，不是每条样本轨迹都相等。

## 思考题
1. 为什么高斯噪声可以合并？
2. $q(x_t\mid x_0)$ 和 $q(x_t\mid x_{t-1})$ 有何区别？
3. 若 $\alpha_1=0.9,\alpha_2=0.8$，则 $\bar\alpha_2=0.72$，请写出 $x_2$ 的等效加噪表达式。
