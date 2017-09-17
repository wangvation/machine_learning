# Linear Regression

## 最小二乘法
什么是最小二乘法？先看下百度百科的介绍：最小二乘法（又称最小平方法）是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳函数匹配。利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其他一些优化问题也可通过最小化能量或最大化熵用最小二乘法来表达。

通过这段描述可以看出来，最小二乘法也是一种优化方法，求得目标函数的最优值。并且也可以用于曲线拟合，来解决回归问题。难怪《统计学习方法》中提到，回归学习最常用的损失函数是平方损失函数，在此情况下，回归问题可以著名的最小二乘法来解决。看来最小二乘法果然是机器学习领域做有名和有效的算法之一。
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

多元线性回归模型自变量与因变量之间为线性关系可表示为：<br/>
$$y=w_0+w_1x_{i1}+w_2x_{i2}+\cdots+w_nx_{in}$$
令
$$w=\left(
 \begin{matrix}
 w_0\\
 w_1\\
 \vdots\\
 w_n
 \end{matrix}
  \right),
  x_i=\left(
 \begin{matrix}
 1\\x_{i1}\\x_{i2}\\ \vdots\\x_{in}
 \end{matrix}
  \right)$$
则上边的式子可以写成如下形式：
$$y=w^Tx_i$$
其中，$w$表示模型的参数，$x_i$表示数据集中第$i$条数据。
损失函数可以写为：
$$L=\frac{1}{N}\sum_{i=1}^n(y_i−w^Tx_i)^2（1）$$

那么，令
$$
 X=\left(
 \begin{matrix}
   x_{1}^T\\
   x_{2}^T\\
   \vdots  \\
   x_{m}^T
  \end{matrix}
  \right)=\left(
 \begin{matrix}
   1 & x_{11} &x_{12} & \cdots &x_{1n} \\
   1 & x_{21} &x_{22} & \cdots &x_{2n} \\
   \vdots & \vdots &\vdots &  & \vdots \\
   1 & x_{m1} &x_{m2} & \cdots &x_{mn}
  \end{matrix}
  \right)
$$
$$
y=\left(
 \begin{matrix}
   y_1\\
   y_2 \\
   \vdots \\
   y_m
  \end{matrix}
  \right),
  w=\left(
 \begin{matrix}
   w_0\\
   w_1 \\
   \vdots \\
   w_n
  \end{matrix}
  \right)
$$

带入（1）式即可得:
$$L=\frac{1}{N}(y−Xw)^T(y−Xw)$$

二.多特征下求解参数 $w$


&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$
L=\frac{1}{N}(y−Xw)^T(y−Xw)\\
=\frac{1}{N}(y^T−w^TX^T)(y−Xw)\\
=\frac{1}{N}(y^Ty−y^TXw−w^TX^Ty+w^TX^TXw)\\
=\frac{1}{N}(w^TX^TXw−2w^TX^Ty+y^Ty)(2)\\
$

我们的目标是让损失函数最小，即求（2）的最小值，我们对$w$求偏导数，令其等于$0$，就可以求出L取得极小值时参数$w$的值。
$$\frac{\partial L}{\partial w}=\frac{1}{N}(2X^TXw−2X^Ty)=0(3)$$
$$⇒X^TXw=X^Ty$$
$$⇒w=(X^TX)^{−1}X^Ty$$
## 梯度下降算法
### 特征缩放
参考链接<br/>
[机器学习经典算法之-----最小二乘法](http://www.cnblogs.com/iamccme/archive/2013/05/15/3080737.html)<br/>
[机器学习笔记（二）——多变量最小二乘法](http://blog.csdn.net/chunyun0716/article/details/50759532)