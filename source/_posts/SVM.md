---
title: SVM
date: 2018-09-27 10:23:10
mathjax: true
tags: ML
---

#公式解析

##SVM
### SVM基本模型问题
$$\min_{\omega,b}\frac{1}{2}{\left \|  \omega \right \|}^2,  \quad (1)$$
$$s.t.\quad  y_i(\omega^Tx_i + b)\geq 1, \quad i = 1,2,...,m   \quad (2)$$

### 转为拉格朗日函数
$$ L(\omega,b,\alpha) = \frac{\left \|  \omega\right \|^2}{2} + \sum_{i=1}^{m}\alpha_i(1-y_i(\omega^Tx_i+b)) \quad (3)$$
$\alpha=(\alpha_1,\alpha_2,...,\alpha_m)^T$为拉格朗日乘子向量$\alpha_i \geq 0$

### 拉格朗日对偶
原始问题：
 $$\min_{\omega,b} \left [  {\max_{\alpha:\alpha_j\geq0}L(\omega,b,\alpha)}   \right ]  \quad (4)$$

对偶问题：
$$\max_{\alpha:\alpha_j\geq0} \left [  {\min_{\omega,b}L(\omega,b,\alpha)}   \right ]  \quad (5)$$
通过对偶问题解决原始问题：
$$ \min_{\omega,b}L(\omega,b,\alpha) = \min \left [    {\frac{\left \|  \omega\right \|^2}{2} + \sum_{i=1}^{m}\alpha_i(1-y_i(\omega^Tx_i+b))}   \right ]  \quad (6)$$
**对$\omega,b$求导:**

$$\frac{\partial L}{\partial \omega} = 0 \Rightarrow \omega = \sum_{i = 1}^{m}\alpha_iy_ix_i  \quad (7)$$

$$\frac{\partial L}{\partial y} = 0 \Rightarrow 0 = \sum_{i = 1}^{m}\alpha_iy_i  \quad (8)$$
将公式7带入公式6中（注意模的平方公式）：
$$ \min_{\omega,b}L(\omega,b,\alpha) = \frac{1}{2}\left [  \sum_{i = 1}^{m}\alpha_iy_ix_i\right ]^T \left [  \sum_{i = 1}^{m}\alpha_iy_ix_i\right ] + \sum_{i = 1}^m\alpha_i- \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j  \quad (9)$$
加号左边的公式转换为：
$$\left [  \sum_{i = 1}^{m}\alpha_iy_ix_i\right ]^T \left [  \sum_{i = 1}^{m}\alpha_iy_ix_i\right ] =   \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j \quad (10)$$
最后公式9转化为
$$ \min_{\omega,b}L(\omega,b,\alpha) = \sum_{i = 1}^m\alpha_i- \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j  \quad (11)$$

最后得出：
#### 对偶问题
$$\max_a \left [  \sum_{i = 1}^m\alpha_i- \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j \right ]  \quad (12)$$

$$s.t. \sum_{i = 1}^{m}\alpha_iy_i = 0,$$
$$ a_i \geq 0, i = 1,2,...,m$$
给上式加上负号 则由求最大值变为求最小值
$$\min_a \left [ \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j -\sum_{i = 1}^m\alpha_i\right ] \quad (13)$$
$$s.t. \sum_{i = 1}^{m}\alpha_iy_i = 0,$$
$$ a_i \geq 0, i = 1,2,...,m$$
上式有不等式存在，需满足**KKT**条件

$$ \left\{\begin{matrix}
\alpha_i \geq 0 \\ 
y_if(x_i)-1 \geq0 \\
\alpha_i(y_if(x_i)-1 ) = 0
\end{matrix}\right.     \quad (14)$$

**KKT条件的约束和原问题的约束区别是：**
原问题的约束是对可行解的约束，KKT的约束是对最优解的约束

### 引入松弛变量(软间隔最大化)
对于线性不可分的训练数据不等式的约束并不能都成立，这时需要修改间隔最大化，使其成为软间隔最大化。线性不可分意味着某些样本点$(x_i,y_i)$不能满足间隔大于1的约束条件，这时可以引入一个松弛变量$\xi_i \geq 0 $,
$$y_i(\omega^T x_i+b) \geq1-\xi_i  \quad (15)$$
同时，对每个松弛变量$\xi_i$，支付一个代价$\xi_i$,目标函数有原来的$\frac{1}{2}{\left \|  \omega \right \|}^2$变成
$$\frac{1}{2}{\left \|  \omega \right \|}^2 + C\sum_{i=1}^m\xi_i \quad (16)$$
 $C>0$称为<font color="#007947" size=3>惩罚参数</font>,一般由应用决定，$C$值大时对误分类的惩罚增大，$C$值小时对误分类的惩罚减小，
线性不可分SVM的基本问题转化成如下：
$$\frac{1}{2}{\left \|  \omega \right \|}^2 + C\sum_{i=1}^m\xi_i , \quad (17)$$
$$s.t.\quad  y_i(\omega^Tx_i + b)\geq 1- \xi_i, \quad i = 1,2,...,m   \quad (18)$$
$$\xi_i \geq 0  \quad (19)$$
根据拉格朗日乘子法得出新公式：
$$ L(\omega,b,\alpha,\xi,\mu) = \frac{\left \|  \omega\right \|^2}{2} + C\sum_{i=1}^m\xi_i -\sum_{i=1}^{m}\alpha_i(y_i(\omega^Tx_i+b)-1+ \xi_i) - \sum_{i = 1}^m\mu_i \xi_i \quad (20)$$
其中$\alpha_i \geq 0$  $\mu_i \geq 0$
求出$ L(\omega,b,\alpha,\xi,\mu) $对 $w$, $b$,$\xi$,的极小
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{\partial \omega} = w-\sum_{i=1}^m\alpha_iy_ix_i = 0$$
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{\partial b} = -\sum_{i=1}^m\alpha_iy_i = 0$$
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{
\partial \xi} = C-\alpha_i -\mu_i = 0$$
得
$$ w = \sum_{i=1}^m\alpha_iy_ix_i = 0  \quad (21)$$
$$ \sum_{i=1}^m\alpha_iy_i = 0  \quad (22)$$
$$ C-\alpha_i -\mu_i = 0  \quad (23)$$
将式（21）、（22）、（23）代入 式（20），得
$$ \min_{w,b,\xi}L(\omega,b,\alpha,\xi,\mu)  = -\frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j +\sum_{i = 1}^m\alpha_i \quad (24)$$
再对$\min_{w,b,\xi}L(\omega,b,\alpha,\xi,\mu) $ 求$\alpha$的极大，即得**对偶问题**：
$$\max_a \left [  \sum_{i = 1}^m\alpha_i- \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j \right ]  \quad (25) (和式12一样)$$
给上式加上负号 则由求最大值变为求最小值

$$\min_a \left [ \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j x_i^T x_j -\sum_{i = 1}^m\alpha_i\right ] \quad (26)(和式13一样)$$
$$s.t. \sum_{i = 1}^{m}\alpha_iy_i = 0,$$
$$C-\alpha_i -\mu_i = 0,  \quad (27)$$
$$ a_i \geq 0, \quad  i = 1,2,...,m \quad (28)$$
$$ \mu_i \geq 0, \quad  i = 1,2,...,m \quad (29)$$

由式(27),式(28)可得
$$0\leq \alpha \leq  C  \quad (30)$$

式(20)结合**KKT**条件得
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{\partial \omega} = w-\sum_{i=1}^m\alpha_iy_ix_i = 0,$$
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{\partial b} = -\sum_{i=1}^m\alpha_iy_i = 0,$$
$$\frac{\partial L(\omega,b,\alpha,\xi,\mu)}{
\partial \xi} = C-\alpha_i -\mu_i = 0,$$
$$ \alpha_i(y_i(\omega^Tx_i+b)-1+ \xi_i) = 0,  \quad (31)$$
$$ \mu_i \xi_i = 0, \quad (32)$$
$$ y_i(\omega^Tx_i+b)-1 + \mu_i \geq 0, \quad (33)$$
$$ \xi_i \geq 0,\quad  i = 1,2,...,m $$
$$ \alpha_i \geq 0,\quad  i = 1,2,...,m $$
$$ \mu_i \geq 0, \quad  i = 1,2,...,m $$

##### 条件分析

- 当$\xi_i > 0$ 说明点在边界内
- 当$\alpha = 0$时，$\mu = C$,因为$ \mu_i \xi_i  = 0$,所以$ \xi_i = 0$,由式(15)可知 ,$y_i(\omega^T x_i+b) \geq1 $,在超平面外。
- 当$\alpha = C$时，$\mu = 0$,$ \xi_i  \geq 0$,由式(15)可知,$y_i(\omega^T x_i+b) < 1 $,在超平面内。
- 当$0 < \alpha < C$时，$\mu \neq  0$,$ \xi_i = 0$, 由式(15)可知,$y_i(\omega^T x_i+b) =1 $,是支持向量点。

##SMO
上面公式的求解可形式化为求解凸二次规划问题，当训练样本很大时，序列最小最优化(sequential minimal optimization)算法可以比较快速地求解。根据上面的公式推导，**SMO**在此处需要解以下凸二次规划问题：

$$\min_a \left [ \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m\alpha_i \alpha_j y_i y_j K(x_i,x_j) -\sum_{i = 1}^m\alpha_i\right ] \quad (34)$$
$$s.t. \quad \sum_{i = 1}^{m}\alpha_iy_i = 0, \quad (35)$$
$$0\leq \alpha_i \leq  C ,\quad  i = 1,2,...,m  \quad (36)$$
$K$函数为核函数

**SMO**算法基本思路：如果所有变量都满足此最优化问题的**KKT**条件，那么这个问题的最优化的解就得到了。因为**KKT**条件是该优化问题的充分必要条件。否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题，这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。子问题有两个变量，一个是违反**KKT**条件最严重的那一个，另一个由约束问题自动确定。如此，**SMO**算法将原问题不断分解为子问题并对子问题求解，进而达到求解原问题的目的。

由式（35）可知

$$ a_1 = -y_1 \sum_{i = 2}^m \alpha_i y_i $$

如果$a_2$确定,那么$a_1$也随之确定



#### 两个变量的二次规划求解

不失一般性，假设选择的两个变量是$\alpha_1,\alpha_2$ ，其他变量是固定的， **SMO**的最优化问题（34）～（36）的子问题可以写成



$$
\begin{eqnarray*} 
\min_{\alpha_1, \alpha2}  \quad W(\alpha_1,\alpha_2) &=  \frac{1}{2}K_{11} \alpha^2 + \frac{1}{2}K_{22} \alpha^2  + y_1 y_2 K_{12} \alpha_1 \alpha_2 \\
  & -(\alpha_1 + \alpha_2)+y_1 \alpha_1 \sum_{i = 3}^m y_i \alpha_iK_{i1}+y_2 \alpha_2 \sum_{i = 3}^m y_i \alpha_iK_{i2}  
\end{eqnarray*} \quad (37)
$$
$$s.t. \quad \alpha_1y_1 + \alpha_2 y_2  = - \sum_{i = 3}^m y_i \alpha_i = \zeta  \quad (38)$$

$$ 0\leq \alpha_i \leq  C ,\quad  i = 1,2  \quad (39)$$

#### $\alpha_2$的上下界分析

式（37）～（39）两个变量$(\alpha_1,\alpha_2)$可以用下图的二维空间图像表示

![图1.0 二变量优化问题图示](/images/smo_1.jpg)

$(\alpha_1,\alpha_2)$的取值在盒子$[0,C] \times [0,C]$内，式（38）使得$(\alpha_1,\alpha_2)$在平行与盒子$[0,C] \times [0,C]$的对角线上，有$y_1 = y_2$和$y_1 \neq  y_2$两种情况。假设问题（37）～（39）的初始可行解为$\alpha_1^{old},\alpha_2^{old}$,最优解为$\alpha_1^{new},\alpha_2^{new}$,沿着约束方向未经剪辑时$\alpha_2$的最优解为$\alpha_2^{new,unc}$。

##### $y_1 \neq  y_2$ 时

记$\alpha_2$的下界和上届分别为$L,U$，根据图像可知$L$在$L_1$或$L_2$取到， $H$在$H_1$或$H_2$取到。 在$L_2$处$\alpha_2 = 0$，在$L_1$处

$$0 - \alpha_2 = \gamma   \quad (40)$$

因为$y_1 \neq  y_2$ ，有

$$\alpha_1^{old} - \alpha_2^{old} = \gamma   \quad (41)$$

上面两式消去$\gamma$得$\alpha_2 = \alpha_2^{old} - \alpha_1^{old}$。由此，当$y_1 \neq  y_2$时，$\alpha_2$的下界可以表示为



$$ \left\{\begin{matrix}
0,   \quad \gamma > 0 \\ 
\alpha_1^{old} - \alpha_2^{old} , \quad  \gamma < 0 \\
\end{matrix}\right.     \quad (42)$$

可以看到$\alpha_2$的下界实际是由$ \alpha_2^{old} - \alpha_1^{old}$和0的大小来决定的，因此上式可重写为

$$L  = \max \{0,\alpha_2^{old} - \alpha_1^{old} \} \quad(43)$$

按照上面的推倒可以得出$\alpha_2$的上界

$$H  = \max \{C, C - \alpha_1^{old} + \alpha_2^{old} \} \quad(44)$$

##### $y_1=  y_2$ 时

根据上面推倒可以得到类似的下界和上界分别为

$$L  = \max \{0,\alpha_2^{old} + \alpha_1^{old} - C\}  \quad(45)$$

$$H  = \max \{C,\alpha_2^{old} + \alpha_1^{old} \}  \quad(46)$$

#### $\alpha_1$和$\alpha_2$的求解

为了叙述简单定义如下

$$g(x) = \sum_{i = 1}^{m}\alpha_i y_i K(x_i,x) + b \quad (47)$$

$$E_i =g(x_i) - y_i =   \left [ \sum_{j = 1}^{m}\alpha_j y_j K(x_j,x) + b  \right ] - y_i , \quad i = 1,2  \quad (48)$$



> 定理：最优化问题（37）～（39）沿着约束方向未经剪辑时的解是
>
> $$\alpha_2^{new,unc} = \alpha_2^{old} + \frac{y_2 (E_1 - E_2) }{\eta}$$
>
> 其中，$\eta = K_{11} + K_{22} - 2K_{12} = {\left \| \Phi(x_1) - \Phi(x_2) \right \|}^2$
>
> $\Phi(x)$是输入空间到特征空间的映射
>
> 经剪辑后的$\alpha_2$的解是
>
> $$ \left\{\begin{matrix}
> H,   \quad \alpha_2^{new,unc} > H\\ 
> \alpha_2^{new,unc},  \quad    L \leq     \alpha_2^{new,unc}  \leq   H \\
>
> L,  \quad  \alpha_2^{new,unc} < L \\
>
> \end{matrix}\right.     \quad (42)$$
>
> 由$\alpha_2^{new}$求得$\alpha_1^{new}$
>
> $$\alpha_1^{new} = \alpha_1^{old} + y_1 y_2 (\alpha_2^{old } - \alpha_2^{new})$$

证明略



#代码解析

#### smo函数

```python
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    # 循环迭代结束 或者 循环遍历所有alpha后，alphaPairs还是没变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas
```

####  innerL 函数

```python
def innerL(i, oS):
    """innerL
    内循环代码
    Args:
        i   具体的某一行
        oS  optStruct对象

    Returns:
        0   找不到最优的值
        1   找到了最优的值，并且oS.Cache到缓存中
    """

    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
```



#### 第一个a

SMO称选择第一个变量的过程称为外循环。外循环在训练样本中选取违反KKT条件最严重的样本点，并将其对应的变量作为第一个变量$\alpha$，即$y_i g(x_i) - 1 < 0$ (由式14得出)，因为定义了$E_i = g(x_i) - y_i$（式48），且$y_i^2 = 1, E_i y_i = y_i g(x_i) - 1$所违反KKT条件就是$E_i y_i < 0$，找出满足这个条件的第一个$\alpha$

在函数`inneL`中
```python
if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
```
`tol`是容忍因子（tol>0）
**左边**： $y_i  E_i < -tol \Rightarrow y_i g(x_i)  -  1< - tol \Rightarrow y_i g(x_i) < 1 - tol < 1$ ,这些点是落在超平面内且$\alpha$要满足$\alpha = C$，否则就违反KKT条件。但是我们就是要找出违反KKT条件的点 就是要找出这个条件，所以才有$\alpha < C$ 这个“与”条件

**右边**： $y_i  E_i > tol \Rightarrow y_i g(x_i)  -  1<  tol \Rightarrow y_i g(x_i) > 1 + tol > 1$ ,这些点就表示落在超平面外,$\alpha$要满足$\alpha = 0$，但是违反KKT的条件就是$\alpha > 0 $，所以才有$\alpha > 0 $ 这个“与”条件

# 参考资料

[1]. 统计学习方法，李航

[2]. 机器学习实战

[3]. 机器学习，周志华

[4]. [零基础学SVM—Support Vector Machine](https://zhuanlan.zhihu.com/p/24638007)

