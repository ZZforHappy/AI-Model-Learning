# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先学习采样更新与跳步加速，再深入分析非马尔可夫构造、边缘分布不变的证明和网络复用的原因。

## 2. 回顾 DDPM 的正向公式

定义噪声调度参数：

```math
\alpha_t=1-\beta_t,\qquad
\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i
```

正向扩散可以直接写为：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
```

DDPM 训练一个网络来预测噪声：

```math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
```

注意：网络预测的是构造 $`x_t`$ 时的标准化**累计噪声**，而非某一单步添加的噪声。

## 3. 由预测噪声估计原始数据

从正向公式移项：

```math
\sqrt{\bar{\alpha}_t}x_0
=x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
```

因此：

```math
x_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}
{\sqrt{\bar{\alpha}_t}}
```

在生成过程中，真实的 $`\epsilon`$ 不可得，以网络预测值 $`\epsilon_\theta(x_t,t)`$ 代替：

```math
\boxed{
\hat{x}_0(x_t,t)=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}
}
```

**含义：**虽然网络本身输出的是噪声，但我们可以根据当前 $`x_t`$ 和预测噪声得到一个“原始数据估计” $`\hat{x}_0`$。这不是已知的真实原始数据，只是模型的估计。

## 4. 为什么可以跳过时间步

设当前时间步为 $`t`$，目标为更早的 $`s`$，其中 $`0\le s<t`$；定义 $`\bar{\alpha}_0=1`$。

我们已经从当前的 $`x_t`$ 计算出 $`\hat{x}_0`$ 和 $`\hat{\epsilon}=\epsilon_\theta(x_t,t)`$。直观上可以借助正向公式的结构，构造目标时间步 $`s`$ 的状态：

```math
\boxed{
x_s=
\sqrt{\bar{\alpha}_s}\hat{x}_0+
\sqrt{1-\bar{\alpha}_s}\,\hat{\epsilon}
}
```

**这正是 DDIM 在确定性情形下的跳步更新形式。**

例如，假设选取时间步序列 $`1000\to800\to600\to400\to200\to0`$，则每次可跨越原来的多个时间步，而不必执行中间每一步。所选序列仅作示意，实际实现应使用与模型噪声调度匹配的索引。

严格来说，上式并不意味着能够随意改变 DDPM 原本的马尔可夫转移分布。DDIM 通过构造一族不同的（一般可为非马尔可夫的）正向联合过程，保留同样的 $`q(x_t\mid x_0)`$ 边缘分布，从而得到新的采样更新方式。这是它能够继续使用原有噪声预测训练目标的重要原因。

## 5. DDIM 的一般采样公式

为了表达可调节的随机性，对 $`t\to s`$（$`s<t`$）的跳步更新写成：

```math
\boxed{
x_s=
\underbrace{\sqrt{\bar{\alpha}_s}\hat{x}_0}_{\text{原始数据估计}}
+
\underbrace{\sqrt{1-\bar{\alpha}_s-\sigma_{t\to s}^{2}}\,
\epsilon_\theta(x_t,t)}_{\text{预测噪声方向}}
+
\underbrace{\sigma_{t\to s}z}_{\text{额外随机项}}
}
```

其中 $`z\sim\mathcal{N}(0,I)`$，并且跳步方差可由参数 $`\eta`$ 控制：

```math
\boxed{
\sigma_{t\to s}=
\eta\sqrt{\frac{1-\bar{\alpha}_s}{1-\bar{\alpha}_t}}
\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_s}}
}
```

通常取 $`0\le\eta\le1`$；这时根号内非负。上式将 DDIM 更新分成三部分：

- **原始数据估计：** $`\sqrt{\bar{\alpha}_s}\hat{x}_0`$，决定模型估计的干净数据信号。
- **预测噪声方向：** $`\sqrt{1-\bar{\alpha}_s-\sigma_{t\to s}^{2}}\epsilon_\theta(x_t,t)`$，控制目标时间步保留的噪声成分。
- **随机扰动：** $`\sigma_{t\to s}z`$，引入条件采样的不确定性。

### 5.1 确定性 DDIM：eta = 0

当 $`\eta=0`$ 时：

```math
\sigma_{t\to s}=0
```

所以得到：

```math
\boxed{
x_s=
\sqrt{\bar{\alpha}_s}\hat{x}_0+
\sqrt{1-\bar{\alpha}_s}\epsilon_\theta(x_t,t)
}
```

这时给定**相同的初始噪声、网络、时间步序列**，并且网络推理无额外随机性，整个生成轨迹是确定的。

“确定性”**不等于**总是生成同一张图片：不同初始噪声 $`x_T`$ 仍可以带来不同结果。

### 5.2 eta > 0

当 $`\eta>0`$ 时，更新中会额外采样 $`z`$，生成轨迹不再仅由初始噪声确定。

注意：**把 DDPM 的反向更新简单删去随机项，并不等于推导出了 DDIM。** DDIM 使用自己的更新公式和时间步关系。对于相邻原始时间步和特定方差选择，一般形式可与 DDPM 使用的后验方差选择相联系，但跳步情况下不能把两者完全等同。

## 6. DDIM 采样伪代码

下面采用从大到小的时间步序列，例如 $`t_0>t_1>\cdots>t_K=0`$。这里的 $`t_0`$ 对应初始噪声时刻；预先定义 $`\bar{\alpha}_0=1`$。

~~~python
# 伪代码：model 为已经训练好的噪声预测网络
# times 为严格递减的时间步序列，最后一个元素为 0
# alpha_bar[t] 为训练时固定的累计噪声参数
x = randn(shape)  # 初始状态 x_{t_0}

for t, s in zip(times[:-1], times[1:]):
    eps = model(x, t)

    # 利用预测噪声估计原始数据
    x0_hat = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])

    # 随机性的系数
    sigma = (
        eta
        * sqrt((1 - alpha_bar[s]) / (1 - alpha_bar[t]))
        * sqrt(1 - alpha_bar[t] / alpha_bar[s])
    )

    # 跳到更早的时间步 s
    if s == 0:
        x = x0_hat  # 对应 alpha_bar[0] = 1
    else:
        z = randn_like(x) if eta > 0 else 0
        x = (
            sqrt(alpha_bar[s]) * x0_hat
            + sqrt(1 - alpha_bar[s] - sigma**2) * eps
            + sigma * z
        )

# x 是最终生成的数据
~~~

这里是**算法伪代码**；实际实现需要处理 PyTorch 张量、批量时间编码、索引以及数值稳定性。

## 7. DDPM 与 DDIM 的对比

| 对比项 | DDPM（原始逐步采样） | DDIM（$`\eta=0`$） |
| --- | --- | --- |
| 网络输出 | 预测噪声 | 预测噪声 |
| 可否复用相同训练网络 | 可以 | 可以 |
| 采样更新 | 随机高斯条件转移 | 给定初始噪声后确定性更新 |
| 跳过大量时间步 | 原始一步转移公式不直接支持 | 通过 DDIM 更新可跳步 |
| 网络调用次数 | 常按全部时间步执行 | 可只选择部分时间步 |
| 速度 | 通常较慢 | 通常较快，具体取决于步数与实现 |

DDIM 的加速主要来自**减少采样时的网络调用次数**，而不是把噪声预测网络换成一个更小的模型。

## 8. 核心公式与结论

**第一：由预测噪声估计原始数据**

```math
\boxed{
\hat{x}_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}
}
```

**第二：DDIM 确定性跳步（$`s<t`$）**

```math
\boxed{
x_s=
\sqrt{\bar{\alpha}_s}\hat{x}_0+
\sqrt{1-\bar{\alpha}_s}\epsilon_\theta(x_t,t)
}
```

**最重要的理解：**DDPM 训练好的网络可以继续用于 DDIM。DDIM 利用噪声预测结果构造更早时间步的状态，从而允许更稀疏的采样时间表；$`\eta=0`$ 时生成轨迹在固定初始条件下是确定性的。

**进一步的理论解释见后续第 9—14 节。**

## 9. 深入理解：非马尔可夫过程的意义

### 9.1 马尔可夫性质与边缘分布

DDPM 的正向过程是马尔可夫链，满足：

```math
q(x_t\mid x_{t-1},\ldots,x_0)=q(x_t\mid x_{t-1})
```

其联合分布分解为：

```math
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})
```

但是我们训练噪声预测网络时，直接采样的是：

```math
q(x_t\mid x_0)=\mathcal N\left(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right)
```

这只是时间步 $`x_t`$ 的边缘条件分布，并未指定所有时间步如何彼此关联。

举例：如果 $`X\sim\mathcal N(0,1`$) 且 $`Y\sim\mathcal N(0,1`$)，既可以令 $`Y=X`$，也可以让两者相互独立。二者的边缘分布相同，但联合分布不同。

**关键认识：相同的边缘分布，并不意味着相同的联合分布。** DDIM 就利用这一自由度，重新设计时间步之间的关联，而不改变训练时所用的边缘条件分布。

## 10. DDIM 的非马尔可夫联合分布

记 DDIM 构造的正向联合分布为 $`q_\sigma`$。它由终点分布和一系列以 $`x_0`$ 为条件的“逆向”条件分布定义：

```math
\boxed{
q_\sigma(x_{1:T}\mid x_0)
=q_\sigma(x_T\mid x_0)
\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0)
}
```

首先令终点边缘分布与 DDPM 一致：

```math
q_\sigma(x_T\mid x_0)
=\mathcal N\left(\sqrt{\bar{\alpha}_T}x_0,(1-\bar{\alpha}_T)I\right)
```

再定义 $`t\ge2`$ 时的条件转移：

```math
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal N\left(\mu_{\sigma,t}(x_t,x_0),\sigma_t^2 I\right)
```

这里的条件均值需要精心设计，才能使 $`q_\sigma(x_{t-1}\mid x_0`$) 与 DDPM 相同。

### 10.1 推导条件均值

假设在给定 $`x_0`$ 后：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad\epsilon\sim\mathcal N(0,I)
```

于是：

```math
\epsilon=\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}
```

令 $`z\sim\mathcal N(0,I`$) 且与 $`\epsilon`$ 独立，构造：

```math
x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}x_0+
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon+\sigma_t z
```

这里要求 $`0\le\sigma_t^2\le1-\bar{\alpha}_{t-1}`$。将上一式中 $`\epsilon`$ 的表达式代入，就得到：

```math
\boxed{
\mu_{\sigma,t}(x_t,x_0)
=\sqrt{\bar{\alpha}_{t-1}}x_0+
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}
}
```

这就是 DDIM 非马尔可夫构造中的核心均值公式。

## 11. 证明：为什么边缘分布保持不变？

验证第 10 节构造的 $`x_{t-1}`$，在给定 $`x_0`$ 后仍服从 DDPM 的边缘分布。

**第一步：条件均值。** 因为 $`E[\epsilon]=E[z]=0`$，所以：

```math
E[x_{t-1}\mid x_0]=\sqrt{\bar{\alpha}_{t-1}}x_0
```

**第二步：条件协方差。** 两个噪声独立，交叉项为零：

```math
\begin{aligned}
\mathrm{Cov}(x_{t-1}\mid x_0)
&=(1-\bar{\alpha}_{t-1}-\sigma_t^2)I+\sigma_t^2 I\\
&=(1-\bar{\alpha}_{t-1})I
\end{aligned}
```

**第三步：分布。** 两个独立高斯变量的线性组合仍然是高斯，因此：

```math
\boxed{
q_\sigma(x_{t-1}\mid x_0)
=\mathcal N\left(\sqrt{\bar{\alpha}_{t-1}}x_0,
(1-\bar{\alpha}_{t-1})I\right)
}
```

从 $`t=T`$ 开始递归，就能证明全部时间步的边缘分布与 DDPM 相同。这说明不同的合法 $`\sigma_t`$ 可以改变联合依赖关系，却不会改变对应的边缘分布。

## 12. 从构造公式得到 DDIM 采样公式

生成时没有真实的 $`x_0`$ 和 $`\epsilon`$，但是可以通过训练好的网络得到：

```math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
```

并估计：

```math
\hat{x}_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}
```

把第 10 节中的真实量用估计值替换：

```math
\boxed{
x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0+
\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)+\sigma_t z
}
```

推广到所选时间步 $`s<t`$ 的跳步更新：

```math
\boxed{
x_s=\sqrt{\bar{\alpha}_s}\hat{x}_0+
\sqrt{1-\bar{\alpha}_s-\sigma_{t\to s}^2}
\epsilon_\theta(x_t,t)+\sigma_{t\to s}z
}
```

令 $`\sigma_{t\to s}=0`$，可得确定性的 DDIM：

```math
\boxed{
x_s=\sqrt{\bar{\alpha}_s}\hat{x}_0+
\sqrt{1-\bar{\alpha}_s}\epsilon_\theta(x_t,t)
}
```

这里的“非马尔可夫”描述的是**正向联合过程**；生成时并不需要知道真实的 $`x_0`$，只需网络计算出的估计值。

## 13. DDIM 为什么能复用 DDPM 网络？

DDPM 的简单噪声预测目标为：

```math
L_{\mathrm{simple}}=
\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]
```

其训练输入由下式直接构造：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
```

而 DDIM 保持：

```math
\boxed{q_\sigma(x_t\mid x_0)=q(x_t\mid x_0)}
```

因此，**常用的简单噪声预测训练目标保持不变**，已有 DDPM 噪声预测网络可直接用于 DDIM 采样。不过，这不意味着两个过程的完整 ELBO 逐项相等。

## 14. 本次深入学习总结

- DDPM 的正向过程是马尔可夫的，而 DDIM 可以构造不同的、一般非马尔可夫的联合过程。
- 不同联合过程可以保留相同的边缘条件分布 $`q(x_t\mid x_0`$)。
- DDIM 的条件均值由标准化累计噪声的表达式构造；通过计算均值与协方差，可以证明边缘分布保持不变。
- 用预测的 $`\epsilon_\theta`$ 和估计的 $`\hat{x}_0`$ 替换真实量，得到采样更新公式。
- 由于简单训练损失只用到了这些相同的边缘条件分布，DDIM 可以复用 DDPM 的网络。

**下一步：**进一步理解参数 $`\eta`$ 与 DDPM 原始随机采样的联系。

上一章：[03 DDPM](03-DDPM.md) ｜ 下一章：[05 Score Matching 与 SDE](05-Score-SDE.md)。
