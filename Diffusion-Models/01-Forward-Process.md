# 01 正向扩散（Forward Diffusion）

## 1. 基本思想

正向扩散是一个人为设计的加噪过程：从原始数据出发，每一步加入独立的高斯噪声，最终使数据分布接近标准高斯分布。正向过程本身不需要训练神经网络。

设原始数据为 $x_0$，扩散总步数为 $T$。给定噪声日程 $\beta_t$，满足：

$$
0 < \beta_t < 1
$$

定义：

$$
\alpha_t = 1-\beta_t
$$

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

其中连乘符号表示：

$$
\bar{\alpha}_t = \alpha_1\alpha_2\cdots\alpha_t
$$

因此：

$$
\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1}
$$

## 2. 单步加噪公式

每一步按以下公式加噪：

$$
x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

这里 $\epsilon_t$ 是标准高斯噪声，各步噪声相互独立：

$$
\epsilon_t \sim \mathcal{N}(0,I)
$$

给定 $x_{t-1}$ 后，只有噪声项是随机的，因此条件均值为：

$$
E[x_t \mid x_{t-1}] = \sqrt{\alpha_t}x_{t-1}
$$

条件协方差为：

$$
\mathrm{Cov}(x_t \mid x_{t-1}) = (1-\alpha_t)I
$$

所以单步转移服从高斯分布：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

由于下一步只依赖当前状态，这构成一个马尔可夫过程。

## 3. 展开前两步

第一步：

$$
x_1 = \sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1
$$

第二步：

$$
x_2 = \sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2
$$

把第一步代入第二步：

$$
x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1)+\sqrt{1-\alpha_2}\epsilon_2
$$

展开并合并信号系数：

$$
x_2 = \sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2
$$

现在把后两项记为 $z$：

$$
z = \sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2
$$

两个独立高斯变量的线性组合仍是高斯变量，且均值为零：

$$
E[z] = 0
$$

利用独立性，协方差等于两项协方差之和：

$$
\mathrm{Cov}(z) = \alpha_2(1-\alpha_1)I+(1-\alpha_2)I
$$

整理：

$$
\mathrm{Cov}(z) = (1-\alpha_1\alpha_2)I
$$

因此，$z$ 与下面的随机变量具有相同分布：

$$
\sqrt{1-\alpha_1\alpha_2}\epsilon
$$

其中 $\epsilon$ 是新的标准高斯随机变量。于是，给定 $x_0$ 时，$x_2$ 的分布可由下式直接采样：

$$
x_2 = \sqrt{\alpha_1\alpha_2}x_0+\sqrt{1-\alpha_1\alpha_2}\epsilon
$$

注意：这里的等号表示我们可以用右边构造一个**同分布的样本**，不表示两次独立采样得到的数值必然相等。

## 4. 推广到任意时刻

假设第 $t-1$ 步可以直接写为：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon'
$$

这里的表达式同样用于构造条件分布相同的样本，$\epsilon'$ 是与新噪声 $\epsilon_t$ 独立的标准高斯噪声。

代入单步加噪公式：

$$
x_t = \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon')+\sqrt{1-\alpha_t}\epsilon_t
$$

展开信号部分：

$$
x_t = \sqrt{\alpha_t\bar{\alpha}_{t-1}}x_0+\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon'+\sqrt{1-\alpha_t}\epsilon_t
$$

后两项仍是独立高斯噪声的线性组合，其协方差为：

$$
[\alpha_t(1-\bar{\alpha}_{t-1})+(1-\alpha_t)]I
$$

化简：

$$
(1-\alpha_t\bar{\alpha}_{t-1})I
$$

利用连乘关系：

$$
\alpha_t\bar{\alpha}_{t-1} = \bar{\alpha}_t
$$

所以合并后的噪声协方差为：

$$
(1-\bar{\alpha}_t)I
$$

最终得到正向扩散最重要的闭式采样公式：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中 $\epsilon$ 服从标准高斯分布。严格地说，上式表示给定 $x_0$ 时的一种等价采样方式。

因此：

$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

**结论：**已知原始数据 $x_0$，可以直接采样任意时刻的 $x_t$，不必真正执行前面的所有加噪步骤。

## 5. 为什么最后会接近标准高斯分布

如果噪声日程使得：

$$
\bar{\alpha}_T \approx 0
$$

那么原始数据前面的系数接近零，噪声前面的系数接近一。因此：

$$
q(x_T \mid x_0) \approx \mathcal{N}(0,I)
$$

这里是概率分布的近似，而不是每条样本轨迹严格相等。

## 6. 核心结论

单步加噪：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

任意时刻的闭式分布：

$$
q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
$$

连乘定义：

$$
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

下一章：[02 反向扩散](02-Reverse-Process.md)。
