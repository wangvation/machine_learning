# Linear Regression
## 最小二乘法
最小二乘法（又称最小平方法）是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳函数匹配。利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其他一些优化问题也可通过最小化能量或最大化熵用最小二乘法来表达。

通过这段描述可以看出来，最小二乘法也是一种优化方法，求得目标函数的最优值。并且也可以用于曲线拟合，来解决回归问题。难怪《统计学习方法》中提到，回归学习最常用的损失函数是平方损失函数，在此情况下，回归问题可以著名的最小二乘法来解决。最小二乘法是机器学习领域做有名和有效的算法之一。
### 一元线性模型
我们先以最简单的一元线性模型来解释最小二乘法。什么是一元线性模型呢？ 监督学习中，如果预测的变量是离散的，我们称其为分类（如决策树，支持向量机等），如果预测的变量是连续的，我们称其为回归。回归分析中，如果只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。对于二维空间线性是一条直线；对于三维空间线性是一个平面，对于多维空间线性是一个超平面...

对于一元线性回归模型, 假设从总体中获取了n组观察值$(X_1,Y_1),(X_2,Y_2),\cdots,(X_n,Y_n)$。对于平面中的这$n$个点，可以使用无数条曲线来拟合。要求样本回归函数尽可能好地拟合这组值。综合起来看，这条直线处于样本数据的中心位置最合理。 选择最佳拟合曲线的标准可以确定为：使总的拟合误差（即总残差）达到最小。有以下三个标准可以选择：

1. 用“残差和最小”确定直线位置是一个途径。但很快发现计算“残差和”存在相互抵消的问题。
2. 用“残差绝对值和最小”确定直线位置也是一个途径。但绝对值的计算比较麻烦。
3. 最小二乘法的原则是以“残差平方和最小”确定直线位置。用最小二乘法除了计算比较方便外，得到的估计量还具有优良特性。这种方法对异常值非常敏感。

最常用的是普通最小二乘法（ Ordinary  Least Square，OLS）：所选择的回归模型应该使所有观察值的残差平方和达到最小。（$Q$为残差平方和）- 即采用平方损失函数。

样本回归模型:
$$Y_i=\hat{\beta_0}+\hat{\beta_1}X_i+e_i$$
 $$\Rightarrow e_i=Y_i-\hat{\beta_0}-\hat{\beta_1}X_i$$
  其中$e_i$为样本($X_i$,$Y_i$)的误差<br/>
平方损失函数：
 $$Q=\sum_{i=1}^{n}e_i^2=\sum_{i=1}^{n}(Y_i-\hat{Y_i})^2=\sum_{i=1}^n(Y_i-\hat{\beta_0}-\hat{\beta_1}X_i)^2$$
通过对$Q$求极值来确定参数$\hat{\beta_0},\hat{\beta_1}$,对$Q$求偏导：<br/>
$$\left\{\begin{matrix}
\frac{\partial Q}{\partial \hat{\beta_0}}=2\sum_{i=1}^{n}(Y_i-\hat{\beta_0}-\hat{\beta_1}X_i)(-1)=0\\
\frac{\partial Q}{\partial \hat{\beta_1}}=2\sum_{i=1}^{n}(Y_i-\hat{\beta_0}-\hat{\beta_1}X_i)(-X_i)=0
\end{matrix}\right.$$
解得：<br/>
$$\hat{\beta_0}=\frac{n\sum X_iY_i- \sum X_i\sum Y_i}{n\sum X_i^2-(\sum X_i)^2}$$
$$\hat{\beta_1}=\frac{\sum {X_i}^2\sum Y_i-\sum X_i\sum X_iY_i}{n\sum {X_i}^2-(\sum X_i)^2}$$
以上是一元线性回归模型的最小二乘法的推导
### 多元线性回归模型
与一元线性回归模型的自变量只有一个特征不同，多元线性回归模型的自变量由多个特征。那么我们就引入特征的向量来表示，这里涉及到矩阵的乘法，向量，矩阵求导等一些线性代数的知识。

多元线性回归模型自变量与因变量之间为线性关系可表示为：
$$h_\theta(x_i)=\theta_0+\theta_1x_{i1}+\theta_2x_{i2}+\cdots+\theta_nx_{in}$$
令
$$\theta=\left(\begin{matrix}
 \theta_0\\ \theta_1\\ \vdots\\\theta_n
 \end{matrix}\right),
  x_i=\left(\begin{matrix}
 1\\x_{i1}\\x_{i2}\\ \vdots\\x_{in}
 \end{matrix}\right)$$
则上边的式子可以写成如下形式：
$$h_\theta(x_i)=\theta^Tx_i$$
其中，$\theta$表示模型的参数，$x_i$表示数据集中第$i$条数据。令
$$X=\left(
 \begin{matrix}
   x_{1}^T\\x_{2}^T\\ \vdots  \\x_{m}^T
  \end{matrix}
  \right)=\left(\begin{matrix}
   1 & x_{11} &x_{12} & \cdots &x_{1n} \\
   1 & x_{21} &x_{22} & \cdots &x_{2n} \\
   \vdots & \vdots &\vdots &  & \vdots \\
   1 & x_{m1} &x_{m2} & \cdots &x_{mn}
  \end{matrix}\right)$$
$$y=\left(\begin{matrix}
   y_1\\ y_2 \\ \vdots \\ y_m
  \end{matrix}\right)$$
则线性模型的表达式可写为：
$$h_\theta(X)=X\theta$$
损失函数可以写为：
$$\begin{align*}L&=\sum_{i=1}^m(h_\theta(x_i)-y_i)^2\\
&=(h_\theta(X)-y)^T(h_\theta(X)-y)\\
&=(X\theta-y)^T(X\theta-y)\end{align*}$$

二.多特征下求解参数 $\theta$:
$$\begin{align*}L&=(X\theta-y)^T(X\theta-y)\\
&=(\theta^TX^T-y^T)(X\theta-y)\\
&=(\theta^TX^TX\theta−\theta^TX^Ty−y^TX\theta+y^Ty)\\
&=(\theta^TX^TX\theta−2\theta^TX^Ty+y^Ty)(2)\end{align*}$$

我们的目标是让损失函数最小，即求（2）的最小值，我们对$\theta$求偏导数，令其等于$0$，就可以求出L取得极小值时参数$\theta$的值。
$$\begin{align*}\frac{\partial L}{\partial \theta}&=(2X^TX\theta−2X^Ty)=0(3)\\
&⇒X^TX\theta=X^Ty\\
&⇒\theta=(X^TX)^{−1}X^Ty\end{align*}$$

推导过程中用到的部分公式:

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$f(\theta)$  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; | &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\frac{\partial f}{\partial \theta}$&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;
:-:|:-:
$\theta^Tx$ | $x$
$x^T\theta$ | $x$
$\theta^T\theta$|$2\theta$
$\theta^TC\theta$|$2C\theta$

## 梯度下降法
在进行线性回归模型参数估计时，最小二乘法与梯度下降法两种方法本质相同。都是在给定已知数据（自变量和因变量）的前提下对因变量算出一个一般性的估值函数。然后对给定新数据的因变量进行估算。两种方法的目标相同，都是在已知数据的框架内，使得估算值与实际值的总平方差尽量更小。，估算值与实际值的总平方差的公式为：
$$\Delta=\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$
其中$h_\theta$为估值函数，$x^{(i)}$为第$i$组数据的自变量，$y^{(i)}$ 为第$i$组数据的因变量，$\theta$为系数向量。

不同的是最小二乘法是直接对$\Delta$求导找出全局最小，是非迭代法。而梯度下降法是一种迭代法，先给定一个$\theta$ ，然后向$\Delta$下降最快的方向调整$\theta$ ，在若干次迭代之后找到局部最小。梯度下降法的缺点是到最小点的时候收敛速度变慢，并且对初始点的选择极为敏感，实际操作中一般是随机选择初始点。采用最小均方法LMS(Least Mean Square)更新$\theta$。下面是具体推导过程：

我们将线性模型的表达式写为：
$$h_\theta(x)=\theta_0+\theta_1x_{1}+\theta_2x_{2}+\cdots+\theta_nx_{n}$$
为了方便推导，将损失函数写为如下形式：
$$J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$
$x^{(i)}$表示数据集中第$i$个数据的自变量，$y^{(i)}$表示数据集中第$i$个数据的因变量，
$h_\theta(x^{(i)})$表示已知的假设函数,$m$为训练集的数量。

对$J(\theta)$求偏导：
$$\frac{\partial J}{\partial \theta_j}=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j,j=(1,2,\cdots,n)$$

更新参数$\theta$:
$$\theta_j:=\theta_j-\alpha\frac{\partial J}{\partial \theta_j}=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$$
由于$\theta$每次更新都用到了数据集中全部的数据，因此这种方式叫做批量梯度下降法。
### 特征缩放
在使用梯度下降算法进行模型参数估计时，训练的样本中会容易出现数据相差很大的情况，这就会造成训练过程中需要经历很长的迭代才能找到最优参数，为了加快梯度下降的执行速度。需要将各个特征的值标准化，使得取值范围大致都在$[-1,1]$区间内。
常用的方法有以下两种：

2. 第一种是0均值标准化 (z-score标准化)
$$x:=\frac{x-u}{\sigma}$$
其中$u$是均值，$\sigma$是标准差。经过处理后的数据符合标准正态分布，即均值为0，标准差为1。

3. 第二种是min-max标准化 (Min-Max Normalization)
 $$x:=\frac{x-min}{max-min}$$
经过处理后数据全部落在$[0,1]区间内$。这种方法有一个缺陷就是当有新数据加入时，可能导致$ma$x和$min$的变化，需要重新定义。

这两种归一化方法的适用场景为：

1. 在不涉及距离度量、协方差计算、数据不符合正太分布的时候，可以使用第一种方法或其他归一化方法。比如图像处理中，将RGB图像转换为灰度图像后将其值限定在[0 255]的范围

2. 在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，第二种方法(Z-score standardization)表现更好。


# 参考链接
[机器学习经典算法之-----最小二乘法](http://www.cnblogs.com/iamccme/archive/2013/05/15/3080737.html)<br/>
[机器学习笔记（二）——多变量最小二乘法](http://blog.csdn.net/chunyun0716/article/details/50759532)<br/>
[详解梯度下降法求解线性模型参数](http://blog.csdn.net/ybdesire/article/details/52895274)<br/>
[数据归一化和两种常用的归一化方法](http://www.cnblogs.com/chaosimple/archive/2013/07/31/3227271.html)<br/>