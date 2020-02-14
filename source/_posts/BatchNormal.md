---
title: BatchNormal
date: 2019-10-15 23:46:01
tags: DL
---
## BatchNormal

BN应用于激活函数之前，可以降低层与层之间一因为底层（左边）参数的微弱变化传到上层（右边）被放大（蝴蝶效应），上层需要不断地去适应这些变化，使得训练变得困难(Internal Convariate Shift)。应用BN后 会使得同批次的训练集的同个特征分布均值为0，方差为1。优点如下：

1. 使得输入数据分布相对稳定，加快学习速度
2. 减小模型对参数敏感性，简化调参，提高训练稳定性
3. 缓解梯度消失问题起到一定正则效果（可以不使用dropout）



公式如下：

<img src="./images/BatchNormal.png" alt="BatchNormal" style="zoom:25%;" />





$\epsilon$是为了防止分母出现0。



#### 求导：

$$\begin{array}{l}{\frac{\partial \ell}{\partial \widehat{x}_{i}}=\frac{\partial \ell}{\partial y_{i}} \cdot \gamma}      \quad (p1).     \\ {\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot\left(x_{i}-\mu_{\mathcal{B}}\right) \cdot \frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-3 / 2}} \quad (p2).   \\ {\frac{\partial \ell}{\partial \mu_{\mathcal{B}}}=\left(\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\right)+\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{\sum_{i=1}^{m}-2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}}     \quad (p3).   \\ {\frac{\partial \ell}{\partial x_{i}}=\frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}+\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}+\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}} \quad (p4). \\ {\frac{\partial \ell}{\partial \gamma}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} \cdot \widehat{x}_{i}} \quad (p5). \\ {\frac{\partial \ell}{\partial \beta}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}}}  \quad (p6)        \end{array}$$



一开始很疑惑公式$p3$到底是怎么来的，不符合复合函数链式法则。复习了高数，查找了相关的资料，发现**复合函数链式法则**和**复合函数的偏导链式法则**   概念是有区别的



#### 复合函数的链式法则

 导数运算法则

  $(1) (u \pm v)^{'}(x) = u^{'}(x) + v{'}(x)$

  $(2) (u v)^{'}(x) = u^{'}(x)v(x) + u(x)v{'}(x)$

  $(2) \frac{u}{v}^{'}(x) = \frac{ u^{'}(x)v(x) - u(x)v{'}(x)}{v^2(x)}$

  设有函数如下：

   $u = g(x), y = f[g(x)]$

 $\frac{\partial y}{ \partial x} = f'(u) \cdot  g'(x)$或者$\frac{\partial y}{ \partial x} = \frac{\partial y}{ \partial u} \cdot  \frac{\partial u}{ \partial x}$



#### 复合函数的偏导 链式法则

若已知二元函数$ z=f(u,v)$，$z$ 是 $u$， $v$ 的函数，但若 $u$ 和 $v$ 都又是 $x$ 和 $y$ 的函数，则 $z$ 最终是 $x$ 和 $y$ 的函数，即

$$z(x, y) = f[u(x,y), v(x,y)]$$

那如何求 $z$ 对 $x$ 和 $y$ 的偏微分呢？我们先来看全微分关系．首先:

$$\mathrm{d} z=\frac{\partial f}{\partial u} \mathrm{d} u+\frac{\partial f}{\partial v} \mathrm{d}v $$

而 $u$ 和 $v$ 的微小变化又都是由 $x$ 和 $y$ 的微小变化引起的

$$\mathrm{d} u=\frac{\partial u}{\partial x} \mathrm{d} x+\frac{\partial u}{\partial y} \mathrm{d} y \quad \mathrm{d} v=\frac{\partial v}{\partial x} \mathrm{d} x+\frac{\partial v}{\partial y} \mathrm{d} y$$

所以得出

$$\begin{aligned} \mathrm{d} z &=\frac{\partial f}{\partial u}\left(\frac{\partial u}{\partial x} \mathrm{d} x+\frac{\partial u}{\partial y} \mathrm{d} y\right)+\frac{\partial f}{\partial v}\left(\frac{\partial v}{\partial x} \mathrm{d} x+\frac{\partial v}{\partial y} \mathrm{d} y\right) \\ &=\left(\frac{\partial f}{\partial u} \frac{\partial u}{\partial x}+\frac{\partial f}{\partial v} \frac{\partial v}{\partial x}\right) \mathrm{d} x+\left(\frac{\partial f}{\partial u} \frac{\partial u}{\partial y}+\frac{\partial f}{\partial v} \frac{\partial v}{\partial y}\right) \mathrm{d} y \end{aligned}$$

$$\begin{aligned} \frac{\partial z}{\partial x} = (\frac{\partial f}{\partial u} \frac{\partial u}{\partial x} + \frac{\partial f}{\partial v} \frac{\partial v}{\partial x})  \\ \frac{\partial z}{\partial y} = (\frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial v} \frac{\partial v}{\partial y})  \end{aligned} $$

这就是 $z$ 关于 $x$ 和 $y$ 的全微分关系．根据定义

$$\begin{array}{l}{\frac{\partial f}{\partial x}=\frac{\partial f}{\partial u} \frac{\partial u}{\partial x}+\frac{\partial f}{\partial v} \frac{\partial v}{\partial x}} \\ {\frac{\partial f}{\partial y}=\frac{\partial f}{\partial u} \frac{\partial u}{\partial y}+\frac{\partial f}{\partial v} \frac{\partial v}{\partial y}}\end{array}$$

也叫偏导的**链式法则**．



根据上边公式证明$p3$



<img src="./images/table3.png" alt="table3" style="zoom:50%;" />

$f$函数为上一层的函数,$\frac{\partial \ell}{\partial y}$为上一层传下来的导数

其中:

$$\mu \sim   \frac{-\mu_{\beta}}{\sqrt{\sigma^{2}+\epsilon}}$$

$$\sigma^2 \sim \sigma^2$$

$$\begin{aligned} \frac{\partial \ell}{ \partial \mu} &= \frac{\partial \ell}{ \partial \widehat{x}} (\frac{\partial \widehat{x}}{ \partial \mu} + \frac{\partial \widehat{x}}{ \partial \sigma^{2}})  \\&= \frac{\partial \ell}{ \partial \widehat{x}} (\frac{\partial \widehat{x}}{ \partial \mu} + \frac{\partial \widehat{x}}{ \partial \sigma^{2}} \frac{\partial \sigma^{2}}{ \partial \mu} ) \\ &= \frac{\partial \ell}{ \partial \widehat{x}} \frac{\partial \widehat{x}}{\partial \mu } + \frac{\partial \ell}{ \partial \sigma^{2}} \frac{\partial \sigma^{2}}{\partial \mu }\end{aligned} $$

   证毕




#### 反向传播

反向传播要求出输入$x$和$\gamma ,\beta$的导数

$\gamma ,\beta$的导数公式$p5, p6$已经给出



**求$\frac{\partial \ell}{ \partial x_i}$:**

由$p1$可$\widehat{x}$导数为：

$${\frac{\partial \ell}{\partial \widehat{x}_{i}}=\frac{\partial \ell}{\partial y_{i}} \cdot \gamma}  \tag{1}$$

$\frac{\partial \ell}{\partial y}$为上一层传下来的导数；



$$\begin{aligned} \frac{\partial \ell}{ \partial x_i} &= \frac{\partial \ell}{ \partial \widehat{x}} (\frac{\partial \widehat{x}}{ \partial x_i} + \frac{\partial \widehat{x}}{ \partial \mu} \frac{\partial \mu}{ \partial x_i}  + \frac{\partial \widehat{x}}{ \partial \sigma^{2}} \frac{\partial \sigma^{2}}{ \partial x_i}) \\ &= \frac{\partial \ell}{ \partial \widehat{x}} \frac{1}{\sqrt{\sigma^2 + \epsilon}}     + (\frac{\partial \ell}{ \partial \widehat{x}} \frac{\partial \widehat{x}}{ \partial \mu}) \frac{\partial \mu}{ \partial x_i} + (\frac{\partial \ell}{ \partial \widehat{x}} \frac{\partial \widehat{x}}{ \partial \sigma^{2}}) \frac{\partial \sigma^{2}}{ \partial x_i} \\ &= \frac{\partial \ell}{ \partial \widehat{x}} \frac{1}{\sqrt{\sigma^2 + \epsilon}}     + \frac{\partial \ell}{ \partial \mu} \frac{1}{m} + \frac{\partial \ell}{ \partial \sigma^{2}}  \frac{\partial \sigma^2}{\partial x_i} \end{aligned} \tag{2}$$



$$\begin{align} \frac{\partial \sigma^2}{\partial x_i} &= \frac{2(x_i - \mu)}{m} \end{align} （\mu在这里是个常数）$$

带入上式

$$\begin{aligned}  \frac{\partial \ell}{ \partial x_i} = \frac{\partial \ell}{ \partial \widehat{x}} \frac{1}{\sqrt{\sigma^2 + \epsilon}}     + \frac{\partial \ell}{ \partial \mu} \frac{1}{m} + \frac{\partial \ell}{ \partial \sigma^{2}}  \frac{2(x_i - \mu)}{m} \end{aligned} \tag{3}$$

接下来要求出$\frac{\partial \ell}{ \partial \mu} , \frac{\partial \ell}{ \partial \sigma^{2}}$:



由上式

$$\begin{aligned} \frac{\partial \ell}{ \partial \mu} &= \frac{\partial \ell}{ \partial \widehat{x}} \frac{\partial \widehat{x}}{\partial \mu } + \frac{\partial \ell}{ \partial \sigma^{2}} \frac{\partial \sigma^{2}}{\partial \mu } \\&= \sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial \ell}{ \partial \sigma^{2}} \sum_{m = j}^m \frac{-2(x_i - \mu)}{m} （\mu在这里是个常数） \\&=   \sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x}} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial \ell}{ \partial \sigma^{2}} \cdot (-2) (\sum_{m = j}^m \frac{x_i}{m} - \sum_{m = j}^m \frac{\mu}{m}) \\ &=  \sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x}} \cdot\frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial \ell}{ \partial \sigma^{2}} \cdot (-2) (\mu - m \cdot \frac{\mu}{m}) \\&=  \sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x}} \cdot\frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \end{aligned} \tag{4}  $$

由$p2$式得出

$$\begin{aligned} \frac{ \partial \ell} {\sigma^2} &=  \sum_{j = m}^m \frac{\partial \ell}{ \partial \widehat{x}} \cdot(-\frac{1}{2}(x_i - \mu)(\sigma^2 + \epsilon))^{-1.5}\\ &=(-\frac{-1}{2})\sum_{j = m}^m (\frac{\partial \ell}{ \partial \widehat{x}} \cdot(x_i - \mu)(\sigma^2 + \epsilon)^{-1.5})  \end{aligned} \tag{5}$$

将4式和5式带入3式

$$\begin{aligned}  \frac{\partial \ell}{ \partial x_i} &= \frac{\partial \ell}{ \partial \widehat{x_i}} \frac{1}{\sqrt{\sigma^2 + \epsilon}}  + \sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x_j}} \cdot\frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \frac{1}{m} + (\frac{-1}{2})\sum_{j = 1}^m (\frac{\partial \ell}{ \partial \widehat{x_j}} (x_i - \mu)(\sigma^2 + \epsilon))^{-1.5} \cdot\frac{2(x_i - \mu)}{m} \\&=  \frac{1}{\sqrt{\sigma^2 + \epsilon}} \frac{\partial \ell}{ \partial \widehat{x_i}}  + ( \frac{-1}{\sqrt{\sigma^2 + \epsilon}}  \frac{1}{m})\sum_{j=1}^m \frac{\partial \ell} {\partial \widehat{x_j}}  + (\frac{-1}{\sqrt{\sigma^2 + \epsilon}} \cdot\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \frac{1}{m}) \sum_{j = m}^m (\frac{\partial \ell}{ \partial \widehat{x_j}}  \frac{(x_j - \mu)}{\sqrt{\sigma^2 + \epsilon}}) \\&= \frac{1}{m\sqrt{\sigma^2 + \epsilon}} \left [ m\frac{\partial \ell}{ \partial \widehat{x_i}} - \sum_{j = 1}^m\frac{\partial \ell}{ \partial \widehat{x_j}} - \widehat{x_i} \sum_{j=1}^m (\frac{\partial \ell}{\partial \widehat{x_j}}\widehat{x_j})\right ]\end{aligned}  \tag{6}$$



#### 总结

$${\frac{\partial \ell}{\partial \widehat{x}_{i}}=\frac{\partial \ell}{\partial y_{i}} \cdot \gamma}  \tag{7}$$

$${\frac{\partial \ell}{\partial \gamma}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} \cdot \widehat{x}_{i}} \tag{8} $$

$$ {\frac{\partial \ell}{\partial \beta}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}}} \tag{9}$$

$$\frac{\partial \ell}{ \partial x_i} =\frac{1}{m\sqrt{\sigma^2 + \epsilon}} \left [ m\frac{\partial \ell}{ \partial x_i} - \sum_{j = 1}^m\frac{\partial \ell}{ \partial x_j} - \widehat{x_i} \sum_{j=1}^m (\frac{\partial \ell}{\partial \widehat{x_j}} \widehat{x_j})\right ] \tag{10}$$







参考:

https://arxiv.org/pdf/1502.03167.pdf

http://wuli.wiki//online/PChain.html

https://kevinzakka.github.io/2016/09/14/batch_normalization/

https://zhuanlan.zhihu.com/p/34879333