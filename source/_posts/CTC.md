---
title: CTC
date: 2020-01-05 15:15:00
tags: DL
---
CTC是序列标注问题中的一种损失函数。

传统序列标注算法需要每一时刻输入与输出符号完全对齐。而CTC扩展了标签集合，添加空元素。

ctc用于训练阶段

## CTC loss

要点：

1. 矩阵 𝛼(前向变量)用于计算loss
2.  矩阵𝛽 (后向变量)用来方便计算gradients.



符号表示：

1. $y_k^t$代表输出序列在第t步输出为**k字符**的概率，举个简单的例子：当输出的序列为$(a-ab-)$时，$y_a^3$ 代表了在第3步输出的字母为a的概率；(下面的例子用apple 加_ 即 a, p, l ,e, _,五个字符举例， 但是映射表中对应的字符会有多个)

2. $p(\pi | x)$代表了给定输入$x$，输出路径为 $\pi$ 的概率；

   由于假设在每一个时间步输出的label的概率都是相互独立的，那么 $p(\pi | x)$用公式来表示为 :

   

   $$p(\pi | x) = \prod_{t=1}^{T}(y_k^t) \tag{1}$$

   

3. $\mathscr{F}$ 代表一种多对一的映射，将输出路径 $\pi$ 映射到 标签序列 $l$ 的一种变换，举个简单的例子 $\mathscr{F}(a-ab-) = \mathscr{F}(-aa--abb) = aab$（其中-代表了空格)

4. $p(l \mid  x)$代表了给定输入$x$，输出为序列$l$的概率:

   因此输出的序列为 $l$ 的概率可以表示为所有输出的路径 $\pi$ 映射后的序列为 $l$ 的概率之和，用公式表示为:

   

    $$p(l | x) = \sum_{\pi \subseteq {\mathscr{F}^{-1}(l)}} p(\pi|x) \tag{2}$$

   

   其中$\pi \subseteq {\mathscr{F}^{-1}}(l) $ 表示在路径有多个路径$\pi$ 对应相同的输出序列$l$,$\mathscr{F}(\pi) = l$



### 前向后向算法（训练阶段）

向前算法要解决的就是对真实输出序列的所有路径概率求和（公式2），直接暴力计算$p(l\mid x)$的复杂度非常高，作者借鉴HMM的Forward-Backward算法思路，利用动态规划算法求解。

为了更形象表示问题的搜索空间，用X轴表示时间序列， Y轴表示输出序列，并把输出序列做标准化处理，输出序列中间和头尾都加上blank，用$l$表示最终标签，$l'$表示扩展后的形式，则由$2|l| + 1 = 2|l’|$，比如$l=apple \Rightarrow l'= \_a\_p\_p\_l\_e\_$



为了理解向前算法， 可以构建表格来理解，将真实标签序列$l$转换为$l'$作为纵坐标（由上至下增大），横轴为时间序列（由左至右增大）。

约束条件：

1. 转换只能往右下方向，其他方向不允许
2. 相同的字符之间起码要有一个空字符
3. 非空字符不能跳过
4. 起点必须从前两个字符开始
5. 终点必须落在结尾两个字符

![ctc_1](/images/ctc_9.png)

相关细节可以参考https://xiaodu.io/ctc-explained/



#### 向前传播公式：

**符号表示**：

$ seq(s)$： 纵轴由上往下第s个字符

$\alpha_t(s) $:   t时刻经过节点s的全部前缀路径的概率总和

1. 当seq(s)为空符号或seq(s) 等于seq(s-1)时

   $$\alpha_t(s) = (\alpha_{t-1}(s) + \alpha_{t-1}(s-1)) \cdot y_{seq(s)}^t$$

<img src="/images/ctc_blank.png" alt="ctc_blank" style="zoom:40%;" />

<img src="/images/ctc_same.png" alt="ctc_same" style="zoom:40%;" />

1. 否则：

   $$\alpha_t(s) = (\alpha_{t-1}(s) + \alpha_{t-1}(s-1) +\alpha_{t-1}(s-2)) \cdot y_{seq(s)}^t$$

<img src="/images/ctc_other.png" alt="ctc_other" style="zoom:40%;" />

3. 初始：

   $$\alpha_1(1) = y_{\_}^{1}, \alpha_1(2) = y_{seq(2)}^{1}, \alpha_1(s) = 0, \forall s > 2$$

 **CTC Loss**

对于上图apple词汇的例子

$$- \boldsymbol{ln}(p(apple | x)) =  -\boldsymbol{ln}(\alpha_8(10) + \alpha_8(11))$$

通用公式表示为:

$$- \boldsymbol{ln}(p(l | x)) =  -\boldsymbol{ln}(\alpha_T(|l'| - 1) + \alpha_T(|l'|)) \tag{3}$$



#### 向后传播公式：

基本和向前公式一样 ，不过是反方向的

**符号表示**：

$\beta_t(s) $:   t时刻经过节点s的全部后缀路径的概率总和

1. 当seq(s)为空符号或seq(s) 等于seq(s-1)时

   $$\beta_t(s) = (\beta_{t+1}(s) + \beta_{t+1}(s+1)) \cdot y_{seq(s)}^t$$

2. 否则：

   $$\beta_t(s) = (\beta_{t+1}(s) + \beta_{t+1}(s+1) +\beta_{t+1}(s+2)) \cdot  y_{seq(s)}^t $$

3. 初始：

   $$\beta_T(|l'|) = y_{\_}^{T}, \beta_T(|l'| - 1) = y_{seq(|l'| - 1)}^{T}, \beta_T(s) = 0, \forall s < |l'| - 1$$

 **CTC Loss**

对于上图apple词汇的例子

$$- \boldsymbol{ln}(p(apple | x)) =  -\boldsymbol{ln}(\beta_1(1) + \beta_2(2))$$

通用公式表示为:

$$- \boldsymbol{ln}(p(l | x)) =  -\boldsymbol{ln}(\beta_1(1) + \beta_2(2)) \tag{4}$$



#### 向前向后公式结合



在任意t时刻，便利所有的s,即可得到全部路径的总和

$$\boldsymbol{p}(\boldsymbol{l} | \boldsymbol{x})=\sum_{\boldsymbol{s}=1}^{\left|\boldsymbol{l}^{\prime}\right|} \frac{\boldsymbol{\alpha}_{\mathrm{t}}(\boldsymbol{s}) \boldsymbol{\beta}_{\boldsymbol{t}}(\boldsymbol{s})}{\boldsymbol{y}_{l_{\boldsymbol{s}}^{'}}^{t}} \tag{5}$$

**公式中的t是任选的**

(除以 $y_{l_{s}^{'}}^t$ 因为在 $\alpha$ 和 $\beta$ 中乘了两次)。

![ctc_10](/images/ctc_10.png)



## 导数

### 防止参数underflow

在Alex Graves的[论文](http://www.cs.toronto.edu/~graves/icml_2006.pdf)中, 为了防止参数underflow，要对$\alpha_t(s), \beta_t(s)$进行标准化转换

$$C_{t} \stackrel{\text { def }}{=} \sum_{s} \alpha_{t}(s), \quad \quad \hat{\alpha}_{t}(s) \stackrel{\text { def }}{=} \frac{\alpha_{t}(s)}{C_{t}}$$

$$D_{t} \stackrel{\text { def }}{=} \sum_{s} \alpha_{t}(s), \quad \quad \hat{\alpha}_{t}(s) \stackrel{\text { def }}{=} \frac{\alpha_{t}(s)}{D_{t}}$$

Loss 函数也由$- \boldsymbol{ln}(p(l | x)) =  -\boldsymbol{ln}(\alpha_T(|l'| - 1) + \alpha_T(|l'|))$变成了:

$- \boldsymbol{ln}(p(l | x)) =  - \sum_{t = 1}^{t} ln(C_t)$



### Softmax 函数求导

设 $X = [x_1,x_2,...,x_n]$ ，$Y = [y_1, y_2,...,y_n], Y = softmax(X)$ 

$$ y_i = \frac{e^{x_i}}{\sum_{j = 1}e^{x_j}} \tag{6}$$

(1) 当 $i = j$ 时

$$\begin{aligned} \frac{\partial y_{i}}{\partial x_{j}} &=\frac{\partial y_{i}}{\partial x_{i}} \\ &=\frac{\partial}{\partial x_{i}}\left(\frac{e^{x_{i}}}{\sum_{k} e^{x_{k}}}\right) \\ &=\frac{\left(e^{x_{i}}\right)^{\prime}\left(\sum_{k} e^{x_{k}}\right)-e^{x_{i}}\left(\sum_{k} e^{x_{k}}\right)^{\prime}}{\left(\sum_{k} e^{x_{k}}\right)^{2}} \\ &=\frac{e^{x_{i}} \cdot\left(\sum_{k} e^{x_{k}}\right)^{2}}{\left(\sum_{k} e^{x_{k}}\right)^{2}} \\ &=\frac{e^{x_{i}} \cdot\left(\sum_{k} e^{x_{k}}\right)}{\left(\sum_{k} e^{x_{k}}\right)^{2}}-\frac{e^{x_{i}} \cdot e^{x_{i}}}{\sum_{k} e^{x_{k}}} \\ &=\frac{e^{x_{i}} \cdot\left(\sum_{i} \cdot y_{i}\right.}{\sum_{k} e^{x_{k}}} \cdot \frac{e^{x_{i}}}{\sum_{k} e_{k}^{x_{k}}} \\ &=y_{i}\left(1-y_{i}\right) \end{aligned}$$

(1) 当 $i \neq  j$ 时

$$\begin{aligned} \frac{\partial y_{i}}{\partial x_{j}} &=\frac{\partial}{\partial x_{j}}\left(\frac{e^{x_{i}}}{\sum_{k} e^{x_{k}}}\right) \\ &=\frac{\left(e^{x_{i}}\right)^{\prime}\left(\sum_{k} e^{x_{k}}\right)}{\left(\sum_{k} e^{x_{k}}\right)^{2}} \\ &=\frac{0 \cdot\left(\sum_{k} e^{x_{k}}\right)-e^{x_{i}} \cdot e^{x_{j}}}{\left(\sum_{k} e^{x_{k}}\right)^{2}} \\ &=\frac{-e^{x_{i}} \cdot e^{x_{j}}}{\left(\sum_{k} e^{x_{k}}\right)^{2}} \\ &=-\frac{e^{x_{i}}}{\sum_{k} e^{x_{k}}} \cdot \frac{e^{x_{j}}}{\sum_{k} e^{x_{k}}} \\ &=-y_{i} \cdot y_{j} \end{aligned}$$

综上所述：$\frac{\partial y_{i}}{\partial x_{j}}=\left\{\begin{array}{l}{=y_{i}-y_{i} y_{i}},当i = j  \\ {=0-y_{i} \cdot y_{j}} 当i \neq j \end{array}\right.$



### ctc求导

由上面公式3公式5：

$$- \boldsymbol{ln}(p(l | x)) =  -\boldsymbol{ln}(\alpha_T(|l'| - 1) + \alpha_T(|l'|)) \tag{3}$$

$$\boldsymbol{p}(\boldsymbol{l} | \boldsymbol{x})=\sum_{\boldsymbol{s}=1}^{\left|\boldsymbol{l}^{\prime}\right|} \frac{\boldsymbol{\alpha}_{\mathrm{t}}(\boldsymbol{s}) \boldsymbol{\beta}_{\boldsymbol{t}}(\boldsymbol{s})}{\boldsymbol{y}_{l_{\boldsymbol{s}}^{'}}^{t}} (t是任意的) \tag{5}$$

注意到 这里笔记的

x----RNN(x) -->$u_k^t$ ---softmax(u)--->  $y_k^t$   (k表示的是字符，是模型中字母映射表中的一个，$t \subseteq  T$，$T$是rnn输出的序列的个数)

$$\frac{\partial \ln (p(\mathbf{l} | \mathbf{x}))}{\partial y_{k}^{t}}=\frac{1}{p(\mathbf{l} | \mathbf{x})} \frac{\partial p(\mathbf{l} | \mathbf{x})}{\partial y_{k}^{t}} \tag{7}$$

定义$lab(l,k) = \{s: l_s^{'} = k\}$ ， $ lab(l,k)$可能为空, 因为RNN输出的序列的长度T 经过ctc 转换后变为$l$, $l \leqslant  T$
字母表中的字符k 不一定会在ctc映射的文本中。

$$\frac{\partial p(l|x)}{ \partial y_k^t} = \frac{1}{y_k^{t^2}} \sum_{s \in lab(l,k)} \alpha_t(s) \beta_t(x) \tag{8}$$

具体求导如下：

![ctc_partial](/images/ctc_partial.png)



根据前面所说的防止underflow 公式可以转换如下：

$$-\frac{\partial ln(p(l|x))}{ \partial u_k^t}=y_{k}^{t}-\frac{1}{y_{k}^{t} Z_{t}} \sum_{s \in \operatorname{lab}(\mathbf{l}, k)} \hat{\alpha}_{t}(s) \hat{\beta}_{t}(s) \tag{9}$$

其中

$p(l|x) = Z_{t} \stackrel{\text { def }}{=} \sum_{s=1}^{\left|1^{\prime}\right|} \frac{\hat{\alpha}_{t}(s) \hat{\beta}_{t}(s)}{y_{1_{s}^{t}}^{t}} \tag{10}$ 

和右边的

$ \sum_{s \in lab(l,k)} \alpha_t(s) \beta_t(x) \tag{11}$

是不一样的，因为式11 是建立一个大小为【映射表长度， 输出序列场地】的矩阵， ctc映射后的字符之外的设为0，其他的根据$\hat{\alpha}_{t}(s) \hat{\beta}_{t}(s)$累加，而$p(l|x)$是一个常数



具体代码参照 [stanford-ctc](https://github.com/d2rivendell/stanford-ctc)

```python
#params 为输出序列 (n, m) n为映射表中字母表个数， m为输出序列个数
# L 为 L = 2 * 真实词汇长度 + 1
grad = np.zeros(params.shape) # 包含了其所以的字母
ab = alphas * betas #和真实词汇的字母有关
for s in xrange(L):
    # blank
    if s % 2 == 0:
        grad[blank, :] += ab[s, :]
        ab[s, :] = ab[s, :] / params[blank, :]
    else:
        grad[seq[(s - 1) / 2], :] += ab[s, :]
        ab[s, :] = ab[s, :] / (params[seq[(s - 1) / 2], :])
absum = np.sum(ab, axis=0)#常数
grad = params - grad / (params * absum)
```



## Beam search decoding

解码是在 预测阶段，

> 符号表示：
>
> $\rho$:  去除了空格和重复字符的输出标签序列
>
> $\rho^e$:字符串$\rho$的结尾字符
>
> $\hat{\rho}$:字符串$\rho$ 除去结尾$\rho^e$后的字符
>
> $\gamma(\rho, t)$: 在t时刻网络输出的假设序列为$\rho$(已经去除空格和折叠)的概率
>
> $\gamma^{-1}(\rho, t)$:   $t$时刻网络输出blank空字符, 输出标签序列为$\rho$的概率
>
> $\gamma^{+1}(\rho, t)$:   $t$时刻网络输出非空字符,输出标签序列为$\rho$的概率



定义：

$$\gamma(\rho, t) = \gamma^{-1}(\rho, t) + \gamma^{+1}(\rho, t) \tag{12}$$



$$\gamma^{-1}(\rho, t) = \gamma(\rho, t - 1) y_b^t \tag{13}$$

$y_b^t $表示$t$时刻输出blank空字符的概率



#### 疑问：

为什么要区分t时刻的字符是不是空的呢？ 如果是为了折叠和去空格，  为什么不区分t时刻的字符和$\rho^e$相等的情况?

答案当然不是为了折叠和去重。 下面的讨论会给出答案

#### 常规束算法：

常规束算法在每个输出中计算当前的假设路径， 当前假设路径是基于上个假设路径，而且不折叠重复字符和移除空格，算法会选择其中得分最高的几种路径作为当前路径，如下图（alphabet of  $\{\epsilon, a, b\}$ and a beam size of three）：

![beam_search](/images/beam_search.png)

图片[来源](https://distill.pub/2017/ctc/)（下同）

#### 改进束算法：

上面的算法无法 处理多个对齐映射到同一输出这种情况，  如果要处理多个对齐映射到同一输出这种情况, 处理的方式是：不保留束中的对齐列表，而是存储折叠重复字符并移除空格后的输出前缀。

但是，移除空格$\epsilon$会有个问题 如下图中：

 T=2时刻 第二个$[a,a] \Rightarrow [a]$ ，$[a , \epsilon] \Rightarrow [a]$ , 有两种场景去重和移除空格$\epsilon$后都输出路径为$a$, 把该假设输出路径和概率存储起来

 T=3时刻，如果碰到结合的还是$a$，结合上一个输出$a$，岂不是也是$[a,a] \Rightarrow [a]$ ？，看样子好像没有问题，但是在T=2时刻假设输出路径$a$ 的所有未折叠路径中有一个是：$[a , \epsilon] \Rightarrow [a]$  最后一个是空格，假设我们在T=2时刻不作折叠去空格操作，结合T=3时刻的$a$: $[a , \epsilon, a] \Rightarrow [aa]$ , 本来是输出$[aa]$的呀，却因为折叠去空格操作变成输出$[a]$, 因为这个过程中缺失了路径中最后一个为空格的信息。 下图中T=3中$a$结合$a$, 会生成T=4中$[a], [aa]$两个路径

![beam_search2](/images/beam_search_2.png)

为了实现上图的输出效果，需要怎么做呢，如下图：

![beam_search2](/images/beam_search_3.png)



我们只需统计之前以空白标记$\epsilon$结尾的所有路径的概率（位于字符中间的$\epsilon$也要统计）。同样的，如果是扩展到$[a]$，那我们计算的就是不以$\epsilon$结尾的所有路径概率。我们需要跟踪当前输出在搜索树中前两处输出。无论是以ϵ结尾还是不以ϵ结尾，如果我们在剪枝时为每一种假设做好得分排序，我们就能在计算中使用组合分数。



就得出公式:

$$\gamma(\rho, t) = \gamma^{-1}(\rho, t) + \gamma^{+1}(\rho, t) \tag{12}$$

​    $$\gamma^{-1}(\rho, t) = \gamma(\rho, t - 1) y_b^t \tag{13}$$

​           $$\gamma^{+}(\rho, t)=\gamma^{+}(\rho, t-1) y_{\rho^{e}}^{t}+\left\{\begin{array}{l}{\gamma(\hat{\rho}, t-1) y_{\rho^e}^{t}, \quad if\quad \rho^{\mathrm{e}} \neq \hat{\rho}^{e}} \\ {\gamma^{-}(\hat{\rho}, t-1) y_{\rho^e}^{t} ,  if\quad \rho^{\mathrm{e}} == \hat{\rho}^{e}} \tag{14} \end{array}\right.$$

#### 公式分析

1. 上式13中$\gamma(\rho, t - 1) y_b^t$和式14的**加号左边**$\gamma^{+1}(\rho, t - 1) y_{\rho^e}^t$表示了$t$时刻和$t-1$时刻的折叠输出是**一样**的。分别表示：

   $t-1$时刻的折叠字符和 $t$时刻空格组合成的路径 折叠后不变的概率（如上所说需要记录）  和   $t-1$时刻的k字符(非空格)组合而成的折叠字符和 $t$时刻相同的k字符合成的路径折叠后不变的概率。有点拗口😅



2. 式14**加号右边**表示了$t$时刻和$t-1$时刻的折叠输出是**不一样**， 有两种情况，而且是互斥的：

​       1).  当$t-1$时刻的折叠路径(注意是折叠过的)的最后一个字符字符(非空)和$t$时刻的字符不一样时， 肯定会生成不同折叠输出, 例如: 

​      $t-1$时刻:

​       $[a,b] \Rightarrow ab$，   $[ab,\epsilon] \Rightarrow ab$

​       $t$时刻, 不能是$b$ :

​      $[ab, c] \Rightarrow abc$

​       2).  当$t-1$时刻的折叠路径(注意是折叠过的)的最后一个字符字符和$t$时刻的字符一样时（非空）， 只有跟$t-1$时刻的组合字符是空格时才能保证$t$时刻的折叠路径是不一样的, 例如:

 $t-1$时刻:

​          $[ab,e] \Rightarrow ab$  (结合空字符)，    $[ab,b] \Rightarrow ab$ (结合非空字符)

 $t$时刻 只有结合$t-1$时刻的空字符才能有不一样的折叠路径：

​         $ [ab,e,b ]  \Rightarrow [abb]$





```python
#https://github.com/githubharald/CTCDecoder
class BeamEntry:
	"information about one single beam at specific time-step"
  #存储唯一折叠路径的信息 保存着未折叠前最后一个空字符概率，未折叠前最后一个非空字符概率
	def __init__(self):
		self.prTotal = 0 # blank and non-blank 对应公式12
		self.prNonBlank = 0 # non-blank
		self.prBlank = 0 # blank 对应公式13
		self.prText = 1 # LM score 有语言模型时才有用
		self.lmApplied = False # flag if LM was already applied to this beam 有语言模型时才有用
		self.labeling = () # beam-labeling

    
class BeamState:
	"information about the beams at specific time-step"
  # 存储所有折叠后的输出路径
	def __init__(self):
		self.entries = {}

	def norm(self):
		"length-normalise LM score"
		for (k, _) in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

	def sort(self):
		"return beam-labelings, sorted by probability"
    #找出概率最大的前 beam siz个折叠后的输出路径
		beams = [v for (_, v) in self.entries.items()]
		sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
		return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
	"calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
	if lm and not childBeam.lmApplied:
		c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
		c2 = classes[childBeam.labeling[-1]] # second char
		lmFactor = 0.01 # influence of language model
		bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
		childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
		childBeam.lmApplied = True # only apply LM once per beam entry


def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
  # 没有在字典里面管理的，就留个位置给它
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, lm, beamWidth=25):
	"beam search as described by the paper of Hwang et al. and the paper of Graves et al."

	blankIdx = len(classes)
	maxT, maxC = mat.shape

	# initialise beam state
  #没有开始之前 最开始的先默认放个空字符，prBlank的概率肯定是100%
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 1
	last.entries[labeling].prTotal = 1

	# go over all time-steps
	for t in range(maxT):
		curr = BeamState()

		# get beam-labelings of best beams 为了减小计算量，需要减枝，只对前beamWidth个感兴趣
		bestLabelings = last.sort()[0:beamWidth]

		# go over best beams
		for labeling in bestLabelings:

			# 先计算和上个时间节点输出折叠序列相同， 且最后一个字符不为空格的概率 公式14加号左边
			prNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

			# 计算和上个时间节点输出折叠序列相同， 且最后一个字符是空格的概率 公式13
			prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

			# add beam at current time-step if needed
			addBeam(curr, labeling)

			# fill in data
			curr.entries[labeling].labeling = labeling
			curr.entries[labeling].prNonBlank += prNonBlank
			curr.entries[labeling].prBlank += prBlank
			curr.entries[labeling].prTotal += prBlank + prNonBlank
			curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
			curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

			# 先计算和上个时间节点输出折叠序列相同的的概率
			for c in range(maxC - 1):#注意， 已经除去了空字符，c肯定非空
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
        #计算和上个时间节点输出折叠序列不相同的概率 公式14加号右边
				if labeling and labeling[-1] == c:
          #只有结合t-1时刻的空字符才能有不一样的折叠路径
					prNonBlank = mat[t, c] * last.entries[labeling].prBlank
				else:
          # 当前字符和t-1时刻的折叠路径最后一个字符（必定非空）不同时，结合c(非空)就会输出不同的折叠路径
					prNonBlank = mat[t, c] * last.entries[labeling].prTotal

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)
				
				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank += prNonBlank
				curr.entries[newLabeling].prTotal += prNonBlank
				
				# 应用语言模型，如果有的话
				applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	 # sort by probability
	bestLabeling = last.sort()[0] # get most probable labeling

	# map labels to chars
	res = ''
	for l in bestLabeling:
		res += classes[l]

	return res

```





参考：

https://blog.csdn.net/JackyTintin/article/details/79425866

[原论文](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

[动态ppt](https://docs.google.com/presentation/d/12gYcPft9_4cxk2AD6Z6ZlJNa3wvZCW1ms31nhq51vMk/pub?start=false&loop=false&delayms=3000&slide=id.g24e9f0de4f_0_19958)

https://www.cnblogs.com/shiyublog/p/10493348.html#_label2_0

https://xiaodu.io/ctc-explained-part2/

https://distill.pub/2017/ctc/

https://stats.stackexchange.com/questions/320868/what-is-connectionist-temporal-classification-ctc