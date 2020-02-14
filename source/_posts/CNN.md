---
title: CNN
date: 2019-10-15 23:46:00
tags: DL
---
#卷积层 （Convolutional Layer）

### 正向传播

#### 相关概念：

pad：图像边缘填充的像素宽度(用来调整输出矩阵大小)
stride： 步长
depth：图片通道数
 假设input size 为W，filter Size 为F，stride为S， pad为P,  则输出size为$(W-F+2P)/S + 1$,

如果输出值不是整数，说明设置的超参数不对，需重新设置。

![cnn_gif](/images/cnn_gif.gif)



#### Convolution 和 Cross-Correlation的区别

**Cross-Correlation**： 把核与对应的输入数据对应相乘再求和

$$(I \otimes K)_{i j}=\sum_{m=0}^{k_{1}-1} \sum_{n=0}^{k_{2}-1} I(i+m, j+n) K(m, n)  \quad(1)$$

**Convolution**： 把核先反转180度，再作协相关

$$\begin{aligned}(I * K)_{i j} &=\sum_{m=0}^{k_{1}-1} \sum_{n=0}^{k_{2}-1} I(i-m, j-n) K(m, n) \quad(2)  \\ &=\sum_{m=0}^{k_{1}-1} \sum_{n=0}^{k_{2}-1} I(i+m, j+n) K(-m,-n) \quad(3)\end{aligned}$$

![hFqwv](/images/hFqwv.png)

#### 卷积和权重共享

（下图在正向传播时把权重翻转了，但是在阅读相关开源代码后发现，正向传播时即初始化的时候权重并不翻转，只是在反向传播计算时才需要翻转）

![convolution-mlp-mapping](/images/convolution-mlp-mapping.png)



![cnn_gradient_finger](/images/cnn_gradient_finger.png)

为了提高计算效率，卷积操作可以转化为如下的矩阵操作：

**img2col**



![img2col_(1)](/images/img2col_(1).png)

![img2col_(2)](/images/img2col_(2).png)

其他： <https://www.zhihu.com/question/28385679>

### 反向传播

常规CNN的实现方式是ConvLayer , 后面接着ReLU 再后面是PoolLayer。 ConvLayer不需要考虑激活函数，因为ReLU是单独分出来的。

#### 约定

1. $$\delta_{j}^{l}=\frac{\partial C}{\partial z_{j}^{l}} \quad(4)$$
2. $$z_{j}^{l}=\sum_{k} w_{j k}^{l} a_{k}^{l-1}+b_{j}^{l} \quad(5)$$
3. $$a_j^l = \sigma(z_j^l) \quad(6)$$
4. 卷积核的维度为$(k_1^l,k_2^l)$
5. $H^l, W^l$为输入的高度和宽度
6. $x,y$为$L$层行，列下标，$x',y'$为$L+1$层行，列下标

![CNNConV_1](/images/CNNConV_1.png)

（忽略输出H，W大小，此图仅作为理解下标而展示）



####  **正向传播过程**



$$ \xcancel{ z_{x, y}^{l+1}=w^{l+1} * \sigma\left(z_{x, y}^{l}\right)+b_{x, y}^{l+1}=\sum_{m} \sum_{n} w_{m, n}^{l+1} \sigma\left(z_{x-m, y-n}^{l}\right) + b_{x,y}^{l+1} \quad(7) } $$

在正向传播代码实现的时候，weights的初始化和DNN一样，并不会翻转180度，只是在反向传播的时候翻转，所以上式可写成

$$z_{x, y}^{l+1}=w^{l+1} * \sigma\left(z_{x, y}^{l}\right)+b_{x, y}^{l+1}=\sum_{m} \sum_{n} w_{m, n}^{l+1} \sigma\left(z_{x+m, y+n}^{l}\right) + b_{x,y}^{l+1} \quad(7.1)$$

卷积层是没有激活函数，所以上式也可表示为
$$
\begin{align}&z^l_{x,y} = \sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} w_{m,n}^{l} z_{x+m,y+n}^{l-1} + b_{x,y}^{l-1}  & x \in [0,H^l-1], y\in [0, W^l-1]\tag{7.2}\end{align}
$$

####  **开始求解$\delta_{x,y}^l$**

$$ \delta_{x, y}^{l}=\frac{\partial C}{\partial z_{x, y}^{l}}=\sum_{x'} \sum_{y'} \frac{\partial C}{\partial z_{x', y'}^{l+1}} \frac{\partial z_{x', y'}^{l+1}}{\partial z_{x, y}^{l}} = \sum_{x'} \sum_{y^{\prime}} \delta_{x', y'}^{l+1} \frac{\partial(\sum_m \sum_n w_{m, n}^{l+1} \sigma(z_{x' + m, y' + n}^{l})+b_{x', y'}^{l+1})}{\partial z_{x, y}^{l}} \quad(8) $$

当   $$x = x' +  m, y = y' + n  \quad(9)$$   时，

$$\frac{\partial\left(\sum_{m} \sum_{n} w_{m, n}^{l+1} \sigma\left(z_{x' + m, y' + n}^{l}\right)+b_{x', y'}^{l+1}\right)}{\partial z_{x, y}^{l}} \quad(10)$$的值才不为0

此时$$\delta_{x,y}^l = \sum_{x'} \sum_{y'} \delta_{x', y'}^{l+1} w_{m, n}^{l+1} \sigma'\left(z_{x, y}^{l}\right)  \quad(11)$$

由9式可知$m = x -x', n = y - y' \quad(12)$ 带入上式可得

$$\sum_{x^{\prime}} \sum_{y^{\prime}} \delta_{x^{\prime}, y^{\prime}}^{l+1} w_{m, n}^{l+1} \sigma^{\prime}\left(z_{x, y}^{l}\right)=\sum_{x^{\prime}} \sum_{y^{\prime}} \delta_{x', y'}^{l+1} w_{x -x', y-y'}^{l+1} \sigma^{\prime}\left(z_{x, y}^{l}\right) \quad(13)$$

最后得 $$\delta_{x,y}^l=\sum_{x^{\prime}} \sum_{y^{\prime}} \delta_{x', y'}^{l+1} w_{x -x', y-y'}^{l+1} \sigma^{\prime}(z_{x, y}^{l}) \quad(14)$$

因为ConvLayer一般 不包括激活函数，可认为$\sigma^{\prime}(z_{x, y}^{l})  = 1$

接下来证明 
$$
\begin{align}
\delta^l =  p\delta^{l+1} * ROT180^{\circ}w^l 
\end{align} \tag {15}
$$
$p\delta^l$ 这在里表示：

- 正向传播的步长为1，pad为0时，$p\delta^l$ 为$\delta^l$ **外围**加上padding（ 高宽为卷积核高宽减1即$(f_1^{l-1}-1,f_2^{l-1}-1)$  ）后的梯度矩阵（步长为1的时候);

- 正向传播的步长不为1，pad为0时，$p\delta^l$ 为$\delta^l$ 的在**行列间** 插入宽高为 $(S - 1)$的零元素，再在外围填充宽高为$(F-1) $的零元素

$p\delta^l$ 和$\delta^l$ 的关系根据核心宽高 和步长的关系如下图所示：

![反向传播填充dz](/images/反向传播填充dz.png)

- pad不为0时,$\delta^l$在式15 的卷积运算之后需要去掉外围padding

**证明：**

已知$m \in [0, k_1{^l} - 1],n\in [0, k_2{^l} - 1]$

根据12式得$x - x' \in [0, k_1{^l} - 1], y - y'\in [0, k_2{^l} - 1]$

变换一下得： $ x' \in [1 - k_1{^l} + x, x], y' \in [1 -  k_2{^l} + y , y]$

由于$x \in [0, H^l - 1], y \in [0, W^l - 1]$

因此有
$$
\begin{cases}
x' \in [\max(0,x +1 - k_1{^l}),\min(H^l-1,x)] \\
y' \in [\max(0,y +1 - k_2{^l}),\min(\ W^l-1,y)]   \tag {16}
\end{cases}
$$
下面来看一个例子，对于l-1层 $5 \times 5$ 的卷积层，卷积核$3 \times 3$ , 则输出的l层卷积大小为5-3+1=3，也就是$3 \times 3$ , 此时有：
$$
\begin{cases}
x' \in [\max(0, x - 2),\min(2,x)] \\
y' \in [\max(0,y - 2),\min(2,y)]   \tag {17}
\end{cases}
$$


根据公式12的约束
$$
\begin{align}
&\delta^{l}_{0,0} =\delta^{l+1}_{0,0}W^{l+1}_{0,0} &x' \in [0,0],y' \in [0,0] \\
&\delta^{l}_{0,1} =\delta^{l+1}_{0,1}W^{l+1}_{0,0} + \delta^{l+1}_{0,0}W^{l+1}_{0,1} &x' \in [0,0],y' \in [0,1] \\
&\delta^{l}_{0,2} =\delta^{l+1}_{0,2}W^{l+1}_{0,0} + \delta^{l+1}_{0,1}W^{l+1}_{0,1} +\delta^{l+1}_{0,0}W^{l+1}_{0,2} &x' \in [0,0],y' \in [0,2] \\
&\delta^{l}_{1,0} =\delta^{l+1}_{1,0}W^{l+1}_{0,0} + \delta^{l+1}_{0,0}W^{l+1}_{1,0} &x' \in [0,1],y' \in [0,0] \\
&\delta^{l}_{1,1} =\delta^{l+1}_{1,1}W^{l+1}_{0,0} + \delta^{l+1}_{0,1}W^{l+1}_{1,0} +\delta^{l+1}_{1,0}W^{l+1}_{0,1} + \delta^{l+1}_{0,0}W^{l+1}_{1,1} &x' \in [0,1],y' \in [0,1] \\
&\delta^{l}_{1,2} = \sum_{x'} \sum_{y'} \delta^{l+1}_{x',y'}  W^{l+1}_{x - x',y - y'} &x' \in [0,1],y' \in [0,2] \\
&... ... \\
&\delta^{l}_{2,2} = \sum_{x'} \sum_{y'} \delta^{l+1}_{x',y'}  W^{l+1}_{x - x',y - y'} &x' \in [0,1],y' \in [0,2] \\
\end{align}
$$
 等价于以下的卷积
$$
\delta^{l}=\left(
\begin{align}
&0, &&0,&&0,&&0,&&0,&&0,&&0 \\
&0, &&0,&&0,&&0,&&0,&&0,&&0 \\
&0,&&0,&&\delta^{l+1}_{0,0},&&\delta^{l+1}_{0,1},&&\delta^{l+1}_{0,2},&&0,&&0\\
&0,&&0,&&\delta^{l+1}_{1,0},&&\delta^{l+1}_{1,1},&&\delta^{l+1}_{1,2},&&0,&&0\\
&0,&&0,&&\delta^{l+1}_{2,0},&&\delta^{l+1}_{2,1},&&\delta^{l+1}_{2,2},&&0,&&0\\
&0,&&0, &&0,&&0,&&0,&&0,&&0 \\
&0,&&0, &&0,&&0,&&0,&&0,&&0
\end{align}
\right) *
\left(
\begin{array}
aW^{l+1}_{2,2},& W^{l+1}_{2,1},& W^{l+1}_{2,0}\\
W^{l+1}_{1,2},& W^{l+1}_{11},& W^{l+1}_{1,0}\\
W^{l+1}_{0,2},& W^{l+1}_{01},& W^{l+1}_{0,0}\\
\end{array}
\right)
$$


​          即以$W^{l}$ 翻转$180^\circ$ 的矩阵为卷积核在$\delta^{l+1}$ 加上padding=2(卷积核为$3 \times 3$ )的矩阵上做卷积的结果。

证毕



#### **求解$ \frac{\partial C}{\partial w_{m, n}^{l}}$和$ \frac{\partial C}{\partial b_{m, n}^{l}}$**,

$$
\begin{align}
&\frac{\partial C}{\partial w_{m, n}^{l}} = \sum_{x} \sum_{y} \frac{\partial C}{\partial z_{x, y}^{l}} \frac{\partial z_{x, y}^{l}}{\partial w_{m, n}^{l}} \\
&=\sum_{x} \sum_{y} \delta_{x, y}^{l} \frac{\partial(\sum_{m'} \sum_{n'} w_{m', n'}^{l} \sigma(z_{x+m', y+n'}^{l-1})+b_{x, y}^{l})}{\partial w_{m, n}^{l}}\\
&=  
{\sum_{x} \sum_{y} \delta_{x, y}^{l} \sigma(z_{x+m, y+n}^{l-1})}
\end{align} \tag {18}
$$


$$
\begin{align}
&\frac{\partial C}{\partial b_{x, y}^{l}}=\sum_{x} \sum_{y} \frac{\partial C}{\partial z_{x, y}^{l}} \frac{\partial z_{x, y}^{l}}{\partial b_{x, y}^{l}} \\
&=\sum_{x} \sum_{y} \delta_{x, y}^{l} \frac{\partial(\sum_{m'} \sum_{n'} w_{m', n'}^{l} \sigma(z_{x-a^{\prime}, y-b^{\prime}}^{l})+b_{x, y}^{l})}{\partial b_{x, y}^{l}} \\
&= \sum_x\sum_y \delta_{x,y}^{l} = \delta_{x,y}^{l} 

\end{align}  \tag {19}
$$





### 代码解析



按照公式的推导

```python
def conv_backward(dZ, cache):
   """
   :param dZ: (m, n_H, n_W, C)
   :param cache: A_prev, w, b, padding, strides
     A_prev:  (m, n_H_prev, n_W_prev, C_prev)
     w: (f, f, C_prev, C)
     b: (1, 1, 1, C)
   :return: dz_prev
   """
   A_prev, W, b, padding, strides = cache
   m, n_H_prev, n_W_prev, C_prev = A_prev.shape
   f, f, C_prev, C = W.shape
   #根据步长，在行列间填充0
   dZ_padding_inner = _insert_zeros(dZ, strides)
   #根据卷积核宽高，在外围填充0
   dZ_padding_in_out = zero_pad(dZ_padding_inner, pad=(f - 1, f - 1))
   #翻转卷积核
   W_flip = np.flip(W, (0, 1))
   # 因为是计算反向传播，需要交换卷积核心输入和输出通道
   W_flip = np.swapaxes(W_flip, 2, 3)
   # dz已经在行列间以及外围填充0。在这里卷积函数的参数pad和strides区默认的0和1就行了
   dA_prev, _ = conv_forward(A=dZ_padding_in_out, W=W_flip, b=np.zeros((1,1,1,C)))
   #计算dw = A_prev 卷积 dz,   (m, n_H_prev + pad, n_W_prev + pad,, C_prev)
   """
    在这里卷积的维度关系：
    (m, n_H_prev, n_W_prev, C_prev)  *  (f, f, C_prev, C_next)  ->  (m , n_H, n_W, C_next)
   
   求dw, 怎样使得 A * dZ -> W 即 (m, n_H_prev, n_W_prev, C_prev)  *  (m , n_H, n_W, C_next)   ？-> (f, f, C_prev, C_next)
   A交换一次C_prev和m， dZ要交换两次m和n_H换， m再和n_W换， 得出的dW需要再交换                                                  
   (C_prev, n_H_prev, n_W_prev, m)  *  ( n_H, n_W, m, C_next)   ？-> (C_prev, f, f , C_next)
   """
   A_prev_swap = np.swapaxes(A_prev, 0, 3)
   dZ_swap = np.swapaxes(dZ, 0, 1)
   dZ_swap = np.swapaxes(dZ_swap, 1, 2)
   dw, _ = conv_forward(A=A_prev_swap, W=dZ_swap, b=np.zeros((1, 1, 1, C)), padding=padding, strides=strides)
   dw = np.swapaxes(dw, 0, 2)
   db = np.sum(np.sum(np.sum(dZ, axis=2, keepdims=True), axis=1, keepdims=True), axis=0, keepdims=True)  # 在高度、宽度上相加；批量大小上相加
   # 如果padding不等于0 在运算过后需要移除padding
   dA_prev = _remove_padding(dA_prev, padding)
   return dA_prev, dw/m, db/m
```





Andrew Ng的教程:

```python

def conv_backward_2(dZ, cache):
    """
    Andrew Ng教程的反向传播
    理解这个函数可以把思想反过来：
    正向传播的时候
    Z[i, h, w, c] = A_pad[i, v_start: v_end,  h_start: h_end, :] * W[:,:,:, c] + b[:, :, :, c]
    反向的时候dA_prev[i, v_start: v_end,  h_start: h_end, :] = Z[i, h, w, c] * W[:,:,:, c]

    这样的好处是， 和正向传播的时候逻辑一致, 加padding就是根据padding加，  stride也是在遍历的时候处理
    不需要考虑给dZ加padding以及W翻转问题

    Arguments:
    dZ --   (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    Returns:
    dA_prev --  (m, n_H_prev, n_W_prev, n_C_prev)
    dW --  (f, f, n_C_prev, n_C)
    db -- g(1, 1, 1, n_C)
    """
    A_prev, W, b, padding, strides = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, padding)
    dA_prev_pad = zero_pad(dA_prev, padding)
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * strides[0]
                    vert_end = vert_start + f
                    horiz_start = w * strides[1]
                    horiz_end = horiz_start + f
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    dA_prev = _remove_padding(dA_prev_pad, padding)
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db
```

