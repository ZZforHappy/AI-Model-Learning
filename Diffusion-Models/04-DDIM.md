# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。


## 9. 深入理解：为什么要引入非马尔可夫过程

### 9.1 DDPM 的马尔可夫性质

DDPM 正向扩散满足：

```math
q(x_t\mid x_{t-1},x_{t-2},\ldots,x_0)
=q(x_t\mid x_{t-1})
```

所以它的联合分布可以写为：

```math
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1})
```

但我们训练噪声预测网络时，通常**直接**从下式构造当前状态：

```math
q(x_t\mid x_0)
=\mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right)
```

训练目标依赖的是每一个时间步的**边缘条件分布** # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q(x_t\mid x_0`$)，而不是整个联合分布中所有时间步之间的关联。

### 9.2 相同的边缘分布，不意味着相同的联合分布

举个简单例子，设 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

X\sim\mathcal{N}(0,1`$)、# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

Y\sim\mathcal{N}(0,1`$)：

- 情况 A：# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

Y=X`$，两者完全相关；
- 情况 B：# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

X`$ 和 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

Y`$ 相互独立。

两种情况下，# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

X`$ 和 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

Y`$ 各自的分布都相同，但二者的联合分布不同。

```math
\boxed{\text{相同的边缘分布}\;\not\Rightarrow\;\text{相同的联合分布}}
```

因此我们有可能重新构造扩散过程中不同时间步的依赖关系，却让各时刻的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q(x_t\mid x_0`$) 与 DDPM 保持一致。这是 DDIM 的重要理论出发点。

## 10. DDIM 如何构造非马尔可夫联合过程

用 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q_\sigma`$ 表示 DDIM 构造的一族正向联合分布。它可以用终点分布和一组“给定原始样本”的反方向条件分布来定义：

```math
\boxed{
q_\sigma(x_{1:T}\mid x_0)
=q_\sigma(x_T\mid x_0)
\prod_{t=2}^{T}q_\sigma(x_{t-1}\mid x_t,x_0)
}
```

其中先指定：

```math
q_\sigma(x_T\mid x_0)
=\mathcal{N}\left(\sqrt{\bar{\alpha}_T}x_0,(1-\bar{\alpha}_T)I\right)
```

对于 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

t\ge 2`$，令：

```math
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal{N}\left(\mu_{\sigma,t}(x_t,x_0),\sigma_t^2 I\right)
```

关键是如何设计 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\mu_{\sigma,t}`$，才能保证每一时刻的边缘分布仍为 DDPM 的形式。

### 10.1 从标准化累计噪声出发

对固定的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$，假定：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0
+\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
```

给定 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_t,x_0`$ 后，标准化噪声可以表示成：

```math
\epsilon
=\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{\sqrt{1-\bar{\alpha}_t}}
```

用它构造前一个时间步：

```math
\boxed{
x_{t-1}
=\sqrt{\bar{\alpha}_{t-1}}x_0
+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon
+\sigma_t z
}
```

其中 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

z\sim\mathcal{N}(0,I`$) 且与 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon`$ 独立。需满足 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

0\le\sigma_t^2\le1-\bar{\alpha}_{t-1}`$。

将上面的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon`$ 表达式代入，得到：

```math
\boxed{
\mu_{\sigma,t}(x_t,x_0)
=\sqrt{\bar{\alpha}_{t-1}}x_0
+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}
{\sqrt{1-\bar{\alpha}_t}}
}
```

这是 DDIM 所用的条件分布均值，而不是通过简单删除 DDPM 随机项得到的。

## 11. 严格证明：为什么边缘分布保持不变

证明目标：

```math
\boxed{
q_\sigma(x_{t-1}\mid x_0)
=\mathcal{N}\left(
\sqrt{\bar{\alpha}_{t-1}}x_0,
(1-\bar{\alpha}_{t-1})I
\right)
}
```

采用上一节的构造。在固定 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$ 时，# 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon`$ 和 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

z`$ 都是独立标准高斯噪声。

**第一步：条件均值。** 由于 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

E[\epsilon]=E[z]=0`$，得到：

```math
\begin{aligned}
E[x_{t-1}\mid x_0]
&=\sqrt{\bar{\alpha}_{t-1}}x_0\\
&\quad+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,E[\epsilon]\\
&\quad+\sigma_t E[z]\\
&=\sqrt{\bar{\alpha}_{t-1}}x_0
\end{aligned}
```

**第二步：条件协方差。** 利用独立性，交叉协方差项为零：

```math
\begin{aligned}
\mathrm{Cov}(x_{t-1}\mid x_0)
&=(1-\bar{\alpha}_{t-1}-\sigma_t^2)I+\sigma_t^2I\\
&=(1-\bar{\alpha}_{t-1})I
\end{aligned}
```

**第三步：确定分布。** 独立高斯变量的线性组合仍为高斯变量，因此上述均值与协方差就足以确定 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q_\sigma(x_{t-1}\mid x_0`$)。

这一证明可从终点 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

T`$ 开始递归用于所有 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

t\ge2`$，从而保证 DDIM 构造的每个 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q_\sigma(x_t\mid x_0`$) 都与 DDPM 的对应边缘分布相同。

**关键结论：**在满足方差非负等合法性条件时，改变 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\sigma_t`$ 会改变时间步之间的关联，却不会改变上述边缘分布。这并不是说所有联合分布相同。

## 12. 从非马尔可夫构造得到生成公式

在生成阶段，真实的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$ 和 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon`$ 都不可得，但我们有已训练的噪声预测网络：

```math
\hat{\epsilon}=\epsilon_\theta(x_t,t)
```

并可估计：

```math
\hat{x}_0
=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}
{\sqrt{\bar{\alpha}_t}}
```

将构造公式中的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$ 和 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon`$ 分别替换为 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\hat{x}_0`$ 与 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\hat{\epsilon}`$：

```math
\boxed{
x_{t-1}
=\sqrt{\bar{\alpha}_{t-1}}\hat{x}_0
+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,
\epsilon_\theta(x_t,t)
+\sigma_t z
}
```

对选定的非相邻时间步 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

s<t`$，采用相应的跳步方差参数后，可以类似地写为：

```math
\boxed{
x_s
=\sqrt{\bar{\alpha}_s}\hat{x}_0
+\sqrt{1-\bar{\alpha}_s-\sigma_{t\to s}^2}\,
\epsilon_\theta(x_t,t)
+\sigma_{t\to s}z
}
```

当 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\sigma_{t\to s}=0`$ 时，便是确定性 DDIM：

```math
\boxed{
x_s=\sqrt{\bar{\alpha}_s}\hat{x}_0
+\sqrt{1-\bar{\alpha}_s}\epsilon_\theta(x_t,t)
}
```

注意：这里“非马尔可夫”描述的是所构造的**正向联合过程**，并不意味着实际生成时必须知道真实 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$。生成阶段只使用网络估计的 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\hat{x}_0`$。

## 13. 为什么可以直接使用 DDPM 训练好的网络

DDPM 常用的简单噪声预测损失为：

```math
L_{\mathrm{simple}}
=\mathbb{E}_{x_0,t,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]
```

训练时，只需从训练数据得到 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

x_0`$，随机取时间步 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

t`$，再用：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
```

构造训练输入。网络的训练样本分布取决于 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q(x_t\mid x_0`$)。

DDIM 已经保证：

```math
\boxed{
q_\sigma(x_t\mid x_0)=q(x_t\mid x_0)
}
```

因此，DDIM 能够复用基于这类边缘分布训练得到的**同一个简单噪声预测网络**，并在采样时采用不同的转移方式。

这不意味着不同联合过程的**完整 ELBO** 逐项都相同；这里强调的是常用的简单噪声预测训练目标不需要改变。

## 14. 本次深入学习的核心总结

1. **DDPM 正向过程是马尔可夫的**：每一步只依赖前一步。
2. **边缘分布相同不代表联合分布相同**：可以改变时间步之间的依赖关系。
3. **DDIM 重新构造联合过程**：使用 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q_\sigma(x_{t-1}\mid x_t,x_0`$)，并保证全部 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

q_\sigma(x_t\mid x_0`$) 与 DDPM 相同。
4. **边缘分布保持不变的证明**：由高斯线性组合的均值、协方差及正态性得到。
5. **从构造到生成**：用网络预测 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\epsilon_\theta`$ 和估计 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\hat{x}_0`$ 代替不可得的真实量，形成 DDIM 采样公式。
6. **为什么无需重训**：简单噪声预测损失只依赖相同的边缘分布，而不是具体的正向联合依赖关系。

**下一步深入：**理解 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\eta`$ 对随机性的控制，以及 # 04 DDIM：确定性采样与跳步加速

## 1. DDIM 要解决什么问题

DDPM 已经能够从随机高斯噪声生成数据，但典型的原始采样需要按照 $`T,T-1,\ldots,1`$ 的顺序，反复调用噪声预测网络。当 $`T=1000`$ 时，通常需要约 1000 次网络预测，生成速度较慢。

DDIM（Denoising Diffusion Implicit Models）的关键目标是：**不必重训原来的噪声预测网络，也能选择更稀疏的时间步来采样生成**，例如由 1000 步改为 50 步。减少时间步通常减少计算量，但步数过少可能影响生成质量。

本章先讲 DDIM 的更新公式与跳步采样，再深入分析非马尔可夫联合过程、边缘分布不变的证明及网络复用原因。

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

**深入推导：**关于非马尔可夫构造、边缘分布证明与网络复用的解释见后续第 9–14 节。

\eta=1`$ 在相邻步、特定反向方差设定下与 DDPM 采样的联系。

上一章：[03 DDPM](03-DDPM.md) ｜ 下一章：[05 Score Matching 与 SDE](05-Score-SDE.md)。
