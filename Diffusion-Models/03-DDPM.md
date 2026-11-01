# 03 DDPM：训练目标与采样算法

## 1. DDPM 的整体思想

DDPM（Denoising Diffusion Probabilistic Models）包含两个过程：

- **正向扩散**：按照预先设定的噪声日程，把真实数据逐步变成近似标准高斯噪声，不需要训练。
- **反向生成**：学习每一步的条件概率分布，从随机噪声逐步采样得到新数据。

设扩散总步数为 $`T`$，定义：

```math
\alpha_t=1-\beta_t,\qquad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
```

其中 $`\beta_t`$ 是标准 DDPM 中预先设定的噪声日程，而不是神经网络学习的参数。

正向一步的分布：

```math
q(x_t\mid x_{t-1})=\mathcal{N}(\sqrt{\alpha_t}x_{t-1},\beta_t I)
```

利用高斯分布的性质，可直接构造任意时刻的训练样本：

```math
\boxed{x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)}
```

这里的 $`\epsilon`$ 是从 $`x_0`$ **直接构造** $`x_t`$ 所用的标准化累计噪声，并非最后一步加入的单步噪声。

## 2. 神经网络究竟预测什么

使用噪声预测网络：

```math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
```

网络输入为当前含噪数据 $`x_t`$ 和时间步 $`t`$，输出与 $`x_t`$ 形状相同的噪声估计。网络通常**不直接输出**最终生成的图片。

训练时，$`x_0`$ 来自训练集；$`t`$ 由我们抽样；$`\epsilon`$ 由我们随机生成；因此 $`x_t`$ 和监督标签 $`\epsilon`$ 都是已知的。网络输入却只有 $`(x_t,t)`$，并不会拿到 $`x_0`$ 或真实 $`\epsilon`$。

使用简单均方误差损失：

```math
\boxed{L_{\mathrm{simple}}(\theta)=\mathbb{E}_{x_0,t,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]}
```

每次训练迭代的步骤：

1. 从数据集中采样 $`x_0`$；
2. 均匀采样 $`t\in\{1,\ldots,T\}`$，并生成 $`\epsilon\sim\mathcal{N}(0,I)`$；
3. 用闭式公式计算 $`x_t`$；
4. 将 $`(x_t,t)`$ 输入网络，得到 $`\epsilon_\theta(x_t,t)`$；
5. 计算噪声预测误差，反向传播，更新网络参数 $`\theta`$。

训练时并不需要沿正向马尔可夫链逐步执行 $`T`$ 次加噪。

**一个细节：**仅从 $`(x_t,t)`$ 通常无法唯一确定生成该 $`x_t`$ 的 $`\epsilon`$，因为原始样本 $`x_0`$ 未知。对于均方误差目标，理论最优预测是条件期望 $`\mathbb{E}[\epsilon\mid x_t,t]`$。

## 3. 为什么训练噪声预测网络，等价于参数化反向分布

反向生成过程建模为：

```math
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\big(\mu_\theta(x_t,t),\sigma_{t}^{2} I\big)
```

对于标准固定方差的 DDPM，$`\sigma_{t}^{2}`$ 预先指定（常见选择为 $`\beta_t`$ 或后验方差 $`\tilde{\beta}_t`$），网络主要通过预测噪声来确定均值。

先回顾已知 $`x_0`$ 的正向后验：

```math
q(x_{t-1}\mid x_t,x_0)=\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t I)
```

其中：

```math
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
```

```math
\tilde{\mu}_t=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
```

由正向闭式采样公式反解：

```math
x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
```

将其代入 $`\tilde{\mu}_t`$，并利用 $`\bar{\alpha}_t=\alpha_t\bar{\alpha}_{t-1}`$，整理得：

```math
\boxed{\tilde{\mu}_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)}
```

在生成阶段，真实 $`\epsilon`$ 不可得，因此以网络预测替换它：

```math
\boxed{\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)}
```

因此存在清晰的关系：

**预测噪声 $`\epsilon_\theta`$ → 计算均值 $`\mu_\theta`$ → 与固定方差 $`\sigma_{t}^{2}I`$ 一起确定反向高斯分布。**

这就是“DDPM 看起来在预测噪声，本质上是在学习反向概率转移”的数学含义。需要区分：这里学习的是**每一步** $`p_\theta(x_{t-1}\mid x_t)`$，而不是直接预测最终数据分布 $`p_\theta(x_0)`$。

## 4. DDPM 为什么引入 ELBO

DDPM 的生成联合分布是：

```math
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
```

其中 $`p(x_T)=\mathcal{N}(0,I)`$。我们希望最大化训练样本的对数似然 $`\log p_\theta(x_0)`$，但需要对全部中间隐变量 $`x_{1:T}`$ 积分：

```math
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}
```

这个积分通常难以直接计算。于是引入已知的正向分布：

```math
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})
```

将其乘除进积分，并使用 Jensen 不等式：

```math
\begin{aligned}
\log p_\theta(x_0)
&=\log\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]\\
&\geq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)\right]
\end{aligned}
```

右边称为**证据下界（ELBO）**。最大化 ELBO，相当于最小化其负值，称作变分损失（VLB）。

利用马尔可夫结构及 Bayes 公式，负 ELBO 可分解为：

```math
L_{\mathrm{VLB}}=L_T+\sum_{t=2}^{T}L_{t-1}+L_0
```

其中以下各项均对正向分布取适当期望：

```math
L_T=D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)
```

```math
L_{t-1}=D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)
```

```math
L_0=-\log p_\theta(x_0\mid x_1)
```

$`L_T`$ 约束正向终点接近高斯先验；中间 KL 项使模型反向分布接近已知 $`x_0`$ 时的正向后验；$`L_0`$ 是数据重建项。以上书写省略了 $`L_T`$、$`L_{t-1}`$、$`L_0`$ 外层的期望记号以突出结构。

## 5. 从 KL 散度推导噪声 MSE

重点观察中间一项。已知：

```math
q(x_{t-1}\mid x_t,x_0)=\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t I)
```

```math
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(\mu_\theta,\sigma_{t}^{2} I)
```

两个各向同性高斯分布之间的 KL 散度，除了均值差以外，还包含方差相关项。由于 $`\tilde{\beta}_t`$、$`\sigma_{t}^{2}`$ 均不依赖 $`\theta`$，对网络参数有关的部分可写成：

```math
L_{t-1}=\frac{1}{2\sigma_{t}^{2}}\|\tilde{\mu}_t-\mu_\theta\|_2^2+C_t
```

$`C_t`$ 表示与 $`\theta`$ 无关的项。利用上一节两种均值的表达式：

```math
\tilde{\mu}_t-\mu_\theta
=
\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}
\left(\epsilon_\theta(x_t,t)-\epsilon\right)
```

所以：

```math
\boxed{L_{t-1}=
\frac{\beta_t^2}{2\sigma_{t}^{2}\alpha_t(1-\bar{\alpha}_t)}
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2+C_t}
```

这说明：**固定反向方差时，最小化高斯分布之间的 KL 散度，在网络参数相关部分上就是最小化加权噪声预测误差。**

标准 DDPM 经常使用进一步简化的损失：

```math
\boxed{L_{\mathrm{simple}}=
\mathbb{E}_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]}
```

它省略了时间步相关权重，并不严格等于原始完整 VLB；但实践中是一种有效、常用的训练目标。

## 6. 训练好的模型如何采样生成

训练完成后，冻结网络参数，从标准高斯分布开始：

```math
x_T\sim\mathcal{N}(0,I)
```

对 $`t=T,T-1,\ldots,1`$，先预测噪声：

```math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
```

再计算反向均值：

```math
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}\right)
```

最后从模型给出的高斯分布中随机采样：

```math
\boxed{x_{t-1}=\mu_\theta(x_t,t)+\sigma_{t} z,\qquad z\sim\mathcal{N}(0,I)}
```

在通常的 DDPM 采样实现中，**最后一步 $`t=1`$ 取 $`z=0`$**，避免输出时额外加入随机扰动。

合并得到常用的单步采样公式（$`t>1`$）：

```math
\boxed{x_{t-1}=
\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)
+\sigma_{t} z}
```

为什么“去噪”还要加 $`\sigma_{t}z`$？因为 DDPM 学习的是**随机概率转移**：给定 $`x_t`$，下一步可能有多种合理的 $`x_{t-1}`$。$`\mu_\theta`$ 决定主要更新方向，$`\sigma_{t}z`$ 体现条件分布中的随机性。不能简单地把每一步的随机项全部删掉就称作 DDIM；DDIM 有专门的采样推导。

### 采样伪代码

~~~python
# model 已训练好；alpha、alpha_bar、beta、sigma 均已确定
x = randn(shape)             # x_T ~ N(0,I)

for t in range(T, 0, -1):
    predicted_noise = model(x, t)

    mean = (
        x - beta[t] / sqrt(1 - alpha_bar[t]) * predicted_noise
    ) / sqrt(alpha[t])

    if t > 1:
        x = mean + sigma[t] * randn_like(x)
    else:
        x = mean              # 最后一步不再加随机项

# x 是生成的数据 x_0
~~~

这是表达算法结构的伪代码；实际 PyTorch 实现还需要处理时间编码、批维度、张量设备、参数索引等细节。

## 7. 训练与采样的区别

| 阶段 | 输入来源 | 神经网络输出 | 是否有真实噪声标签 | 是否更新网络 |
| --- | --- | --- | --- | --- |
| 训练 | 真实 $`x_0`$ 加噪构造 $`x_t`$ | $`\epsilon_\theta(x_t,t)`$ | 有，人工采样的 $`\epsilon`$ | 是 |
| 采样 | 从 $`x_T\sim\mathcal{N}(0,I)`$ 反向迭代 | $`\epsilon_\theta(x_t,t)`$ | 没有 | 否 |

训练时知道 $`x_t`$ 是如何构造的，因而可以监督噪声预测；生成时没有真实图片，也没有真实噪声标签，只能让网络不断预测反向分布并逐步采样。

## 8. 三个必须记住的公式

**正向构造训练数据：**

```math
\boxed{x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon}
```

**训练噪声预测网络：**

```math
\boxed{L_{\mathrm{simple}}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2}
```

**反向随机采样（$`t>1`$）：**

```math
\boxed{x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sigma_{t}z}
```

**核心理解：**DDPM 并不是一次性输出新数据。它通过噪声预测网络**间接确定每一步反向高斯分布的均值**，配合预设方差完成连续的随机采样，最终从噪声生成数据。

下一章：[04 DDIM 采样](04-DDIM.md)。
