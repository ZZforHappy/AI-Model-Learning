# 02 反向扩散（Reverse Diffusion）

## 1. 反向扩散要解决什么问题

正向扩散从真实数据 $x_0$ 出发，逐步加入独立高斯噪声，最终使 $x_T$ 接近标准高斯噪声。反向扩散希望从随机噪声 $x_T$ 出发，一步一步得到数据 $x_0$。

正向过程为：

```math
x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
```

这里：

```math
\alpha_t=1-\beta_t
```

```math
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
```

```math
\epsilon_t\sim\mathcal{N}(0,I)
```

上一章得到任意时刻的正向分布：

```math
q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
```

反向扩散真正需要的是：

```math
q(x_{t-1}\mid x_t)
```

不能简单把正向加噪公式移项，因为生成时不知道正向过程实际加入的噪声 $\epsilon_t$。同一个 $x_t$ 可能来自多种不同的 $x_{t-1}$。

因此，反向过程是概率推断问题。

## 2. 为什么先求带有 x0 的后验

直接计算 $q(x_{t-1}\mid x_t)$ 很困难，因为真实数据分布未知。我们先假定还知道原始数据 $x_0$，计算：

```math
q(x_{t-1}\mid x_t,x_0)
```

根据贝叶斯公式和正向过程的马尔可夫性质：

```math
q(x_{t-1}\mid x_t,x_0)
=
\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
```

在固定 $x_t$ 和 $x_0$ 时，分母不随 $x_{t-1}$ 变化，所以：

```math
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
```

我们已经知道右侧的两个分布：

```math
q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1},\beta_t I)
```

```math
q(x_{t-1}\mid x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I)
```

下面对 $t\geq 2$ 推导。$t=1$ 时，给定 $x_0$ 后，后验退化为集中在 $x_0$ 的点质量分布。

## 3. 将高斯概率密度写成指数形式

对于均值 $\mu$、协方差 $\sigma^2 I$ 的高斯分布，其概率密度为：

```math
p(x)=\frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{\|x-\mu\|^2}{2\sigma^2}\right)
```

省略与 $x_{t-1}$ 无关的系数，第一个分布为：

```math
q(x_t\mid x_{t-1})
\propto
\exp\left(-\frac{\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2}{2\beta_t}\right)
```

第二个分布为：

```math
q(x_{t-1}\mid x_0)
\propto
\exp\left(-\frac{\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2}{2(1-\bar{\alpha}_{t-1})}\right)
```

两者相乘时，指数相加：

```math
q(x_{t-1}\mid x_t,x_0)
\propto
\exp\left(-\frac{1}{2}F(x_{t-1})\right)
```

其中：

```math
F(x_{t-1})=
\frac{\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2}{\beta_t}
+
\frac{\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2}{1-\bar{\alpha}_{t-1}}
```

## 4. 展开平方项并合并系数

为了减少符号，暂时记 $y=x_{t-1}$。

第一个平方项展开：

```math
\|x_t-\sqrt{\alpha_t}y\|^2
=
\|x_t\|^2-2\sqrt{\alpha_t}x_t^Ty+\alpha_t\|y\|^2
```

第二个平方项展开：

```math
\|y-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
=
\|y\|^2-2\sqrt{\bar{\alpha}_{t-1}}x_0^Ty+\bar{\alpha}_{t-1}\|x_0\|^2
```

将两式代入 $F(y)$，把含有 $y$ 的项收集起来：

```math
F(y)=A\|y\|^2-2b^Ty+C
```

这里 $C$ 与 $y$ 无关，而二次项系数为：

```math
A=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}
```

一次项系数为：

```math
b=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
```

所以后验密度可以写成：

```math
q(y\mid x_t,x_0)\propto\exp\left(-\frac{1}{2}(A\|y\|^2-2b^Ty+C)\right)
```

## 5. 配方得到高斯均值和方差

利用配方恒等式：

```math
A\|y\|^2-2b^Ty
=
A\left\|y-\frac{b}{A}\right\|^2-\frac{\|b\|^2}{A}
```

把与 $y$ 无关的项并入归一化系数，得到：

```math
q(y\mid x_t,x_0)
\propto
\exp\left(-\frac{A}{2}\left\|y-\frac{b}{A}\right\|^2\right)
```

与高斯密度的标准形式对比，可知：

```math
\tilde{\mu}_t=\frac{b}{A}
```

```math
\tilde{\beta}_t=\frac{1}{A}
```

这里的后验协方差是 $\tilde{\beta}_t I$。

## 6. 详细化简后验方差

代入 $A$ 的表达式：

```math
\tilde{\beta}_t=
\frac{1}{\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}}
```

通分：

```math
\tilde{\beta}_t=
\frac{\beta_t(1-\bar{\alpha}_{t-1})}
{\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t}
```

分母展开：

```math
\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t
=
\alpha_t+\beta_t-\alpha_t\bar{\alpha}_{t-1}
```

因为：

```math
\alpha_t+\beta_t=1
```

```math
\alpha_t\bar{\alpha}_{t-1}=\bar{\alpha}_t
```

所以：

```math
\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t=1-\bar{\alpha}_t
```

最终得到：

```math
\boxed{\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}
```

## 7. 详细化简后验均值

因为 $\tilde{\mu}_t=b/A=\tilde{\beta}_t b$，所以：

```math
\tilde{\mu}_t=
\tilde{\beta}_t\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right)
```

代入后验方差：

```math
\tilde{\mu}_t=
\frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right)
```

分别约去两项中能够消去的因子：

```math
\boxed{
\tilde{\mu}_t=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
}
```

至此得到精确的条件后验：

```math
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t I)
}
```

注意：这是**同时给定 $x_t$ 和 $x_0$** 时的后验，不等于实际生成阶段直接需要的 $q(x_{t-1}\mid x_t)$。

## 8. 为什么实际反向生成需要神经网络

生成时我们只有从标准高斯分布采样的初始噪声：

```math
x_T\sim\mathcal{N}(0,I)
```

原始数据 $x_0$ 并不知道。真正需要的反向分布是：

```math
q(x_{t-1}\mid x_t)=\int q(x_{t-1}\mid x_t,x_0)q(x_0\mid x_t)\,dx_0
```

这个积分涉及真实数据分布，通常无法直接求得，并且积分结果不一定是严格的高斯分布。

DDPM 用神经网络定义一个可学习的高斯反向转移：

```math
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
```

其中网络参数为 $\theta$。协方差可以固定，也可以学习。常见的 DDPM 参数化方式是让网络预测噪声，而不是直接预测均值。

## 9. 从预测噪声推导反向均值

由正向扩散的闭式采样公式：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
```

解出 $x_0$：

```math
x_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
```

如果网络预测噪声 $\epsilon_\theta(x_t,t)$，便可构造原始数据的估计：

```math
\hat{x}_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
```

将这个估计代入上面推导的后验均值：

```math
\mu_\theta(x_t,t)=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\hat{x}_0
```

再代入 $\hat{x}_0$ 的表达式。因为：

```math
\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}}=\frac{1}{\sqrt{\alpha_t}}
```

可以把包含 $x_t$ 的项合并为：

```math
\frac{1}{\sqrt{\alpha_t}}x_t
```

而包含预测噪声的项化简为：

```math
-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
```

于是得到 DDPM 中常用的反向均值公式：

```math
\boxed{
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right)
}
```

按照所选择的反向方差 $\sigma_t^2 I$，一步采样可写成：

```math
x_{t-1}=\mu_\theta(x_t,t)+\sigma_t z
```

其中 $z\sim\mathcal{N}(0,I)$。最后一步通常不再添加随机噪声。

## 10. 本章结论

反向扩散首先通过贝叶斯公式得到给定原始数据的精确高斯后验：

```math
q(x_{t-1}\mid x_t,x_0)=\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t I)
```

后验均值和方差通过**高斯密度相乘、展开平方项、配方**得到。

生成阶段原始数据未知，因此需要训练神经网络近似反向过程。DDPM 常用的做法是让网络预测噪声，再由预测噪声计算反向均值。

下一章：[03 DDPM 训练与损失函数](03-DDPM.md)。
