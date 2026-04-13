# 机器学习与深度学习数学基础完全教程

> 本教程从零开始，系统讲解机器学习所需的四大数学模块：线性代数、微积分、概率统计、图论。每节均包含**核心概念 → 定理推导 → 直觉理解 → 代码示例 → 经典应用**。

---

# 目录

- [第一模块：线性代数与矩阵分析](#第一模块线性代数与矩阵分析)
- [第二模块：微积分与优化](#第二模块微积分与优化)
- [第三模块：概率论与数理统计](#第三模块概率论与数理统计)
- [第四模块：图论](#第四模块图论)

---

# 第一模块：线性代数与矩阵分析

## 1.1 基本对象：标量、向量、矩阵、张量

### 概念定义

| 对象 | 维度 | 符号 | 示例 |
|------|------|------|------|
| 标量（Scalar） | 0 | $a \in \mathbb{R}$ | 学习率 $\eta = 0.01$ |
| 向量（Vector） | 1 | $\mathbf{x} \in \mathbb{R}^n$ | 一张 28×28 图展开成 784 维向量 |
| 矩阵（Matrix） | 2 | $A \in \mathbb{R}^{m \times n}$ | 权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ |
| 张量（Tensor） | ≥3 | $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots}$ | 彩色图像 $\in \mathbb{R}^{H \times W \times 3}$ |

### 向量的几何意义

向量 $\mathbf{x} = (x_1, x_2)^T$ 可理解为从原点出发的箭头，其**方向**和**长度**完整描述了它。

**向量加法**（平行四边形法则）：

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \end{pmatrix}$$

**标量乘法**（缩放向量）：

$$c \mathbf{v} = \begin{pmatrix} c v_1 \\ c v_2 \end{pmatrix}$$

### 矩阵乘法

矩阵 $A \in \mathbb{R}^{m \times k}$， $B \in \mathbb{R}^{k \times n}$，乘积 $C = AB \in \mathbb{R}^{m \times n}$：

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

**直觉**： $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列做内积，得到 $C$ 的 $(i,j)$ 元素。

**ML 中的应用**：全连接层的前向传播就是矩阵乘法：

$$\mathbf{h} = W \mathbf{x} + \mathbf{b}$$

其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$， $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$。

```python
import numpy as np

W = np.random.randn(64, 128)   # 权重矩阵
x = np.random.randn(128)        # 输入向量
b = np.zeros(64)                # 偏置

h = W @ x + b                   # 矩阵-向量乘法
print(h.shape)                  # (64,)
```

---

## 1.2 向量空间、内积与范数

### 向量空间

满足加法和标量乘法封闭性的集合称为**向量空间**。例如 $\mathbb{R}^n$ 对通常的加法和实数乘法构成向量空间。

### 内积（点积）

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^{n} u_i v_i$$

几何意义：

$$\mathbf{u}^T \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

- $\theta = 0°$：同向，内积最大（正相关）
- $\theta = 90°$：正交，内积为 0（无相关）
- $\theta = 180°$：反向，内积最小（负相关）

**ML 应用**：注意力机制中，Query 与 Key 的相似度用内积衡量： $\text{score} = \mathbf{q}^T \mathbf{k}$。

### 范数（向量的"长度"）

**L2 范数**（欧氏距离，最常用）：

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**L1 范数**（曼哈顿距离）：

$$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$$

**L∞ 范数**（无穷范数）：

$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

**直觉**：以 $\mathbf{x} = (3, 4)$ 为例：
- $\|\mathbf{x}\|_2 = \sqrt{9+16} = 5$（直线距离）
- $\|\mathbf{x}\|_1 = 3 + 4 = 7$（街区距离）
- $\|\mathbf{x}\|_\infty = 4$（最大坐标）

**ML 应用**：
- L2 范数 → Ridge 正则化 $\lambda \|\mathbf{w}\|_2^2$
- L1 范数 → Lasso 正则化 $\lambda \|\mathbf{w}\|_1$（促进稀疏性）

```python
x = np.array([3.0, 4.0])
print(np.linalg.norm(x, ord=2))   # L2: 5.0
print(np.linalg.norm(x, ord=1))   # L1: 7.0
print(np.linalg.norm(x, ord=np.inf))  # L∞: 4.0
```

---

## 1.3 线性变换与基变换

### 线性变换

函数 $T: \mathbb{R}^n \to \mathbb{R}^m$ 称为线性变换，若满足：

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$（可加性）
2. $T(c\mathbf{u}) = c \cdot T(\mathbf{u})$（齐次性）

**关键结论**：任何线性变换都可以用矩阵乘法表示： $T(\mathbf{x}) = A\mathbf{x}$。

常见线性变换的矩阵表示（二维）：

**旋转 $\theta$ 角：**

$$
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

**水平缩放 $c$ 倍：**

$$
\begin{pmatrix} c & 0 \\ 0 & 1 \end{pmatrix}
$$

**投影到 $x$ 轴：**

$$
\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
$$

### 基变换

标准基 $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$ 只是众多基之一。若选择新基 $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$（以矩阵 $P$ 的列表示），则坐标变换：

$$\mathbf{x}_{\text{新坐标}} = P^{-1} \mathbf{x}$$

**ML 应用**：PCA 就是找到一组新基（主成分），使数据在这组基下方差最大。

---

## 1.4 特征值与特征向量

### 定义

矩阵 $A \in \mathbb{R}^{n \times n}$ 的特征值 $\lambda$ 和特征向量 $\mathbf{v} \neq \mathbf{0}$ 满足：

$$A \mathbf{v} = \lambda \mathbf{v}$$

**直觉**： $\mathbf{v}$ 在 $A$ 的作用下只被拉伸（$|\lambda| > 1$）、压缩（$|\lambda| < 1$）或翻转（$\lambda < 0$），不改变方向。

### 求法

由 $A\mathbf{v} = \lambda\mathbf{v}$ 得 $(A - \lambda I)\mathbf{v} = \mathbf{0}$。

要有非零解，需行列式为零：

$$\det(A - \lambda I) = 0 \quad \text{（特征多项式）}$$

**示例：**

$$
A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}
$$

$$\det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$$

得 $\lambda_1 = 3, \lambda_2 = 2$。代回求特征向量：
- $\lambda_1 = 3$： $(A - 3I)\mathbf{v} = \mathbf{0}$ → $\mathbf{v}_1 = (1, 0)^T$
- $\lambda_2 = 2$： $(A - 2I)\mathbf{v} = \mathbf{0}$ → $\mathbf{v}_2 = (-1, 1)^T$

### 特征分解（对角化）

若 $A$ 有 $n$ 个线性无关特征向量，则：

$$A = Q \Lambda Q^{-1}$$

其中 $Q$ 的列为特征向量， $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$。

**实对称矩阵**的特征分解更优美（谱定理，见 1.7）：

$$A = Q \Lambda Q^T, \quad Q^T Q = I$$

```python
A = np.array([[3, 1], [0, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)        # [3. 2.]
print("特征向量:\n", eigenvectors)   # 列向量
```

---

## 1.5 正定与半正定矩阵

### 定义

实对称矩阵 $A$ 称为：

- **正定**（PD）： $\mathbf{x}^T A \mathbf{x} > 0$，对所有非零向量 $\mathbf{x}$
- **半正定**（PSD）： $\mathbf{x}^T A \mathbf{x} \geq 0$，对所有向量 $\mathbf{x}$

**等价条件**：
- PD $\Leftrightarrow$ 所有特征值 $\lambda_i > 0$
- PSD $\Leftrightarrow$ 所有特征值 $\lambda_i \geq 0$

**ML 意义**：
- 协方差矩阵是半正定的（方差不能为负）
- 海森矩阵正定 $\Rightarrow$ 该点是局部极小值（损失面是碗状）
- Ridge 回归中加 $\lambda I$ 保证正定性，避免矩阵奇异

---

## 1.6 奇异值分解（SVD）

### 定理

任意矩阵 $A \in \mathbb{R}^{m \times n}$（不要求方阵！）都可以分解为：

$$\underline{\boldsymbol{A = U \Sigma V^T}}$$

其中：
- $U \in \mathbb{R}^{m \times m}$：正交矩阵，列为**左奇异向量**
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 为**奇异值**
- $V \in \mathbb{R}^{n \times n}$：正交矩阵，列为**右奇异向量**

### 几何意义

SVD 把矩阵 $A$ 的作用分解为三步：
1. $V^T$：旋转输入空间
2. $\Sigma$：在各方向上缩放
3. $U$：旋转输出空间

### SVD 与特征分解的关系

$$A^T A = V (\Sigma^T \Sigma) V^T \quad \Rightarrow \quad \sigma_i^2 = \lambda_i(A^T A)$$

即 $A$ 的奇异值是 $A^T A$ 的特征值的平方根。

### 低秩近似（Eckart-Young 定理）

**定理**：在所有秩为 $k$ 的矩阵中，截断 SVD 给出最佳近似：

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T \approx A$$

最小化 $\|A - A_k\|_F^2$，误差为 $\sum_{i=k+1}^{\min(m,n)} \sigma_i^2$。

**直觉**：用最重要的 $k$ 个"成分"来近似原矩阵，丢弃小奇异值对应的成分（噪声）。

```python
A = np.random.randn(100, 50)
U, sigma, VT = np.linalg.svd(A, full_matrices=False)

# 保留前 k=10 个奇异值的低秩近似
k = 10
A_k = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]

# 解释方差比例
explained = np.sum(sigma[:k]**2) / np.sum(sigma**2)
print(f"前{k}个奇异值解释 {explained:.1%} 的方差")
```

### 经典应用：推荐系统

用户-物品评分矩阵 $R \in \mathbb{R}^{m \times n}$ 往往极度稀疏。SVD 矩阵分解：

$$R \approx U \Sigma V^T$$

- $U$ 的行向量：用户的"潜在偏好"向量
- $V$ 的行向量：物品的"潜在特征"向量
- 预测未评分项： $\hat{R}_{ij} = \mathbf{u}_i^T \mathbf{v}_j$

---

## 1.7 迹、秩、行列式

### 迹（Trace）

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii} = \sum_{i=1}^{n} \lambda_i$$

**性质**：
- $\text{tr}(AB) = \text{tr}(BA)$（循环不变性）
- $\text{tr}(A^T A) = \sum_{i,j} A_{ij}^2 = \|A\|_F^2$（Frobenius 范数的平方）

### 秩（Rank）

矩阵 $A$ 的秩 = 线性无关的行数 = 线性无关的列数 = 非零奇异值个数。

$$\text{rank}(A) = \{\sigma_i > 0\}$$

**实用判断**：若 $A \in \mathbb{R}^{m \times n}$ 且 $m < n$，则 $\text{rank}(A) \leq m$（行秩限制）。

### 行列式（Determinant）

$$\det(A) = \prod_{i=1}^{n} \lambda_i$$

- $\det(A) \neq 0 \Leftrightarrow A$ 可逆
- $|\det(A)|$ = $A$ 对应线性变换的"体积缩放比例"
- $\det(AB) = \det(A)\det(B)$

---

## 1.8 基本定理

### 谱定理

**定理**：实对称矩阵 $A = A^T$ 一定可以正交对角化：

$$A = Q \Lambda Q^T$$

其中 $Q$ 的列是两两正交的单位特征向量， $\Lambda$ 是特征值构成的对角矩阵。

**意义**：实对称矩阵的特征向量构成 $\mathbb{R}^n$ 的一组正交基，原空间可以沿这些轴分解。

**ML 应用**：协方差矩阵 $\Sigma = \frac{1}{n} X^T X$ 是实对称正半定矩阵，谱定理保证了 PCA 的合法性。

### 瑞利商定理

对实对称矩阵 $A$，**瑞利商**定义为：

$$R(\mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

**定理**：

$$\lambda_{\min} \leq R(\mathbf{x}) \leq \lambda_{\max}$$

且最大值在 $\mathbf{x} = \mathbf{v}_{\max}$（最大特征向量）处取到，最小值在 $\mathbf{x} = \mathbf{v}_{\min}$ 处取到。

**ML 应用**：PCA 第一主成分方向就是协方差矩阵的最大特征向量（最大化数据投影方差）。

---

## 1.9 PCA：主成分分析

### 问题设定

给定数据矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本， $d$ 维特征），找一个方向 $\mathbf{w}$（$\|\mathbf{w}\|=1$），使投影 $X\mathbf{w}$ 的方差最大。

### 推导

中心化数据后，协方差矩阵：

$$C = \frac{1}{n} X^T X \in \mathbb{R}^{d \times d}$$

最大化 $\text{Var}(X\mathbf{w}) = \mathbf{w}^T C \mathbf{w}$（约束 $\|\mathbf{w}\|=1$）

由瑞利商定理，最优解是 $C$ 的最大特征向量 $\mathbf{v}_1$。

前 $k$ 个主成分 = $C$ 的前 $k$ 个特征向量（按特征值降序）。

### SVD 方法（等价且更数值稳定）

对中心化数据矩阵做 SVD： $X = U \Sigma V^T$

主成分即 $V$ 的前 $k$ 列，降维结果为 $Z = X V_k \in \mathbb{R}^{n \times k}$。

```python
from sklearn.preprocessing import StandardScaler

# 生成示例数据
X = np.random.randn(200, 50)   # 200 样本，50 维
X = StandardScaler().fit_transform(X)  # 标准化

# 方法1：特征分解
C = X.T @ X / len(X)
eigenvalues, eigenvectors = np.linalg.eigh(C)
# eigh 专用于对称矩阵，返回升序排列
idx = np.argsort(eigenvalues)[::-1]   # 降序
V_k = eigenvectors[:, idx[:10]]       # 前10个主成分

# 方法2：SVD（推荐）
U, sigma, VT = np.linalg.svd(X, full_matrices=False)
Z = X @ VT[:10].T    # 降到10维

# 解释方差
explained_var = sigma**2 / np.sum(sigma**2)
print(f"前10个PC解释 {np.sum(explained_var[:10]):.1%} 方差")
```

---

## 1.10 最小二乘法与正规方程

### 问题

给定 $X \in \mathbb{R}^{n \times d}$， $\mathbf{y} \in \mathbb{R}^n$，求 $\mathbf{w}$ 使残差平方和最小：

$$\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|_2^2$$

### 推导正规方程

展开目标函数：

$$L(\mathbf{w}) = (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y}) = \mathbf{w}^T X^T X \mathbf{w} - 2\mathbf{y}^T X \mathbf{w} + \mathbf{y}^T\mathbf{y}$$

对 $\mathbf{w}$ 求梯度并令其为零：

$$\frac{\partial L}{\partial \mathbf{w}} = 2 X^T X \mathbf{w} - 2 X^T \mathbf{y} = \mathbf{0}$$

$$\underline{\boldsymbol{X^T X \mathbf{w}} = X^T \mathbf{y}}$$

若 $X^T X$ 可逆： $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$

### Moore-Penrose 伪逆

当 $X^T X$ 奇异（列线性相关）时，使用伪逆：

$$\mathbf{w}^* = X^+ \mathbf{y}, \quad X^+ = V \Sigma^+ U^T$$

其中 $\Sigma^+$ 是将 $\Sigma$ 的非零奇异值取倒数后转置得到的。

### 正则化

**Ridge（L2）**： $\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda\|\mathbf{w}\|_2^2$

正规方程变为： $\mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$

加 $\lambda I$ 确保矩阵可逆，且特征值从 $\sigma_i^2$ 变为 $\sigma_i^2 + \lambda$（数值更稳定）。

**Lasso（L1）**： $\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda\|\mathbf{w}\|_1$

无闭合解，需迭代求解。L1 正则化会产生稀疏解（部分权重恰好为 0）。

---

## 1.11 特征脸（Eigenfaces）

**问题**：人脸图像维度很高（如 $128 \times 128 = 16384$ 维），如何降维？

**方法**：对人脸数据集做 PCA，得到的主成分即"特征脸"（Eigenfaces）。

1. 收集 $n$ 张人脸图像，每张展平为向量 $\mathbf{x}_i \in \mathbb{R}^d$
2. 中心化： $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$
3. 计算协方差矩阵的特征向量（或对数据矩阵做 SVD）
4. 每张人脸 = 平均脸 + 若干特征脸的线性组合
5. 用前 $k$ 个特征脸的系数表示人脸（大幅降维）

---

# 第二模块：微积分与优化

## 2.1 导数、偏导数与梯度

### 一元导数

函数 $f: \mathbb{R} \to \mathbb{R}$ 在 $x_0$ 的导数：

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$

**几何意义**：切线斜率，描述函数在该点的变化率。

### 偏导数

多元函数 $f(x_1, x_2, \ldots, x_n)$ 对 $x_i$ 的偏导数（固定其他变量）：

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i + h, \ldots) - f(\ldots, x_i, \ldots)}{h}$$

### 梯度

梯度是偏导数组成的向量：

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$$

**关键性质**：梯度指向函数增大最快的方向，模长等于该方向的方向导数。

**示例**： $f(x_1, x_2) = x_1^2 + 2x_2^2$

$$\nabla f = (2x_1, 4x_2)^T$$

在 $(1, 1)$ 处： $\nabla f = (2, 4)^T$，指向东偏北方向。

### 方向导数

$f$ 在单位向量 $\mathbf{d}$ 方向上的变化率：

$$D_{\mathbf{d}} f = \nabla f \cdot \mathbf{d} = \|\nabla f\| \cos\theta$$

由此得：梯度方向（$\theta = 0$）使方向导数最大，**负梯度方向**使函数下降最快——梯度下降的理论基础。

---

## 2.2 全微分与雅可比矩阵

### 全微分

多元函数 $f$ 的全微分：

$$df = \frac{\partial f}{\partial x_1} dx_1 + \frac{\partial f}{\partial x_2} dx_2 + \cdots + \frac{\partial f}{\partial x_n} dx_n = (\nabla f)^T d\mathbf{x}$$

### 雅可比矩阵

向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的雅可比矩阵 $J \in \mathbb{R}^{m \times n}$：

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

$$J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$

**ML 应用**：
- Softmax 函数的雅可比矩阵用于反向传播
- $|det(J)|$ 给出变量替换时的体积缩放因子（流模型中重要）

---

## 2.3 海森矩阵

### 定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的海森矩阵 $H \in \mathbb{R}^{n \times n}$：

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

海森矩阵是对称的（若混合偏导连续），捕获了函数的**二阶曲率信息**。

### 二阶泰勒展开

在 $\mathbf{x}_0$ 附近：

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)$$

### 极值判断

在驻点 $\nabla f = \mathbf{0}$ 处：

| 海森矩阵 | 结论 |
|----------|------|
| 正定（所有特征值 $> 0$） | 局部极小值 |
| 负定（所有特征值 $< 0$） | 局部极大值 |
| 不定（特征值有正有负） | 鞍点 |

**ML 应用**：深度学习中，损失函数鞍点远多于极小值；优化算法需要跨越大量鞍点。

---

## 2.4 链式法则

### 一元情形

若 $y = f(u), u = g(x)$，则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

### 多元情形（向量形式）

若 $\mathbf{z} = f(\mathbf{y}), \mathbf{y} = g(\mathbf{x})$，则：

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

即雅可比矩阵相乘。

### 计算图视角

深度神经网络可以看作一个有向无环图（计算图）：

```
x → [层1] → h1 → [层2] → h2 → [损失函数] → L
```

**反向传播**（Backpropagation）就是对计算图从右到左应用链式法则：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

---

## 2.5 反向传播算法

### 两层神经网络手动推导

设网络： $\mathbf{h} = \sigma(W_1 \mathbf{x})$， $\hat{\mathbf{y}} = \text{softmax}(W_2 \mathbf{h})$，损失 $L = \text{CrossEntropy}(\hat{\mathbf{y}}, \mathbf{y})$

**前向传播**：

$$\mathbf{z}_1 = W_1 \mathbf{x}, \quad \mathbf{h} = \sigma(\mathbf{z}_1)$$

$$\mathbf{z}_2 = W_2 \mathbf{h}, \quad \hat{\mathbf{y}} = \text{softmax}(\mathbf{z}_2)$$

$$L = -\sum_j y_j \log \hat{y}_j$$

**反向传播**（链式法则）：

**Step 1**：Softmax + CrossEntropy 的梯度（经典结果）：

$$\frac{\partial L}{\partial \mathbf{z}_2} = \hat{\mathbf{y}} - \mathbf{y}$$

**Step 2**：对 $W_2$ 的梯度：

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \mathbf{h}^T = (\hat{\mathbf{y}} - \mathbf{y}) \mathbf{h}^T$$

**Step 3**：传递到第一层：

$$\frac{\partial L}{\partial \mathbf{h}} = W_2^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{h}} \odot \sigma'(\mathbf{z}_1)$$

**Step 4**：对 $W_1$ 的梯度：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^T$$

### Softmax + 交叉熵梯度推导

**Softmax**： $\hat{y}_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$

**交叉熵**： $L = -\sum_j y_j \log \hat{y}_j$（one-hot 标签，真实类别为 $c$，即 $y_c = 1$，其余为 0）

$$\frac{\partial L}{\partial z_j} = \hat{y}_j - y_j$$

推导过程：

$$\frac{\partial L}{\partial z_j} = -\sum_k y_k \frac{\partial \log \hat{y}_k}{\partial z_j}$$

利用 $\frac{\partial \hat{y}_k}{\partial z_j} = \hat{y}_k(\delta_{kj} - \hat{y}_j)$，代入整理得 $\hat{y}_j - y_j$。

**直觉**：梯度就是"预测概率 - 真实标签"，简洁优美！

---

## 2.6 优化方法

### 梯度下降（GD）

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

- $\eta$：学习率
- 每次用**全部**训练样本计算梯度

**收敛性**（凸函数情形）：若 $L$ 是 $\beta$-光滑凸函数，则：

$$L(\mathbf{w}_T) - L(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|_2^2}{2\eta T}$$

学习率选 $\eta \leq 1/\beta$ 时收敛。

### 随机梯度下降（SGD）

每次随机取一个（或一批）样本计算近似梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L_i(\mathbf{w}_t)$$

**优点**：计算快，可跳出局部极小
**缺点**：梯度估计噪声大，收敛抖动

### 牛顿法

利用二阶信息，更新步长：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1} \nabla L(\mathbf{w}_t)$$

**优点**：收敛快（二阶收敛）
**缺点**：计算 $H^{-1}$ 代价高（$O(n^3)$），深度学习中不实用

### 拟牛顿法（BFGS）

用秩1或秩2矩阵近似海森矩阵的逆，避免精确计算。

$$H_{t+1}^{-1} \approx H_t^{-1} + \text{rank-2 update}$$

### 梯度裁剪

防止梯度爆炸（RNN 中常见）：

```python
# 按范数裁剪
max_norm = 1.0
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

若 $\|\mathbf{g}\| > \text{max\_norm}$,则 $\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\text{max\_norm}}{\|\mathbf{g}\|}$。

---

## 2.7 约束优化

### 拉格朗日乘子法（等式约束）

问题： $\min f(\mathbf{x})$ s.t. $g(\mathbf{x}) = 0$

引入拉格朗日函数：

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

**一阶条件**（KKT for equality）：

$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \lambda \nabla g = \mathbf{0}$$

**几何解释**：最优点处， $f$ 和 $g$ 的梯度平行（目标函数等值线与约束面相切）。

**应用**：PCA 主成分推导中，最大化 $\mathbf{w}^T C \mathbf{w}$ 约束 $\|\mathbf{w}\|^2 = 1$：

$$\mathcal{L} = \mathbf{w}^T C \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

$$\nabla_{\mathbf{w}} \mathcal{L} = 2C\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} \Rightarrow C\mathbf{w} = \lambda\mathbf{w}$$

### KKT 条件（不等式约束）

问题： $\min f(\mathbf{x})$ s.t. $g_i(\mathbf{x}) \leq 0$， $h_j(\mathbf{x}) = 0$

**KKT 条件**（必要条件）：

1. **平稳性**： $\nabla f + \sum_i \mu_i \nabla g_i + \sum_j \lambda_j \nabla h_j = \mathbf{0}$
2. **原始可行性**： $g_i(\mathbf{x}) \leq 0$， $h_j(\mathbf{x}) = 0$
3. **对偶可行性**： $\mu_i \geq 0$
4. **互补松弛**： $\mu_i g_i(\mathbf{x}) = 0$

**互补松弛**含义：要么 $\mu_i = 0$（约束不活跃），要么 $g_i = 0$（约束紧绑定）。

**应用**：SVM 的硬间隔分类器通过 KKT 条件推导出支持向量（只有紧绑约束上的样本 $\mu_i > 0$ 才是支持向量）。

---

# 第三模块：概率论与数理统计

## 3.1 随机变量与概率分布

### 基本概念

**随机变量**：将样本空间映射到实数的函数 $X: \Omega \to \mathbb{R}$。

**离散分布**：用概率质量函数（PMF） $P(X = x)$ 描述，满足 $\sum_x P(X=x) = 1$。

**连续分布**：用概率密度函数（PDF） $p(x)$ 描述，满足 $\int_{-\infty}^{+\infty} p(x) dx = 1$。

### 期望、方差、协方差

**期望**（均值）：

$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \quad \text{（离散）}$$

$$\mathbb{E}[X] = \int x \cdot p(x) dx \quad \text{（连续）}$$

**方差**（离散程度）：

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**协方差**（两变量线性相关程度）：

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

**相关系数**（归一化协方差）：

$$\rho = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} \in [-1, 1]$$

---

## 3.2 常见概率分布

### 伯努利分布（Bernoulli）

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

- 期望： $p$，方差： $p(1-p)$
- **ML应用**：二分类模型的输出（逻辑回归）

### 多项分布（Multinomial）

$n$ 次独立试验， $k$ 个类别，第 $i$ 类概率 $p_i$：

$$P(X_1=k_1, \ldots, X_m=k_m) = \frac{n!}{k_1! \cdots k_m!} p_1^{k_1} \cdots p_m^{k_m}$$

**ML应用**：多分类模型的输出（Softmax + 交叉熵）

### 高斯（正态）分布

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

记为 $X \sim \mathcal{N}(\mu, \sigma^2)$。

**多维高斯**： $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$：

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

**中心极限定理**：大量独立同分布随机变量之和趋向正态分布，这是高斯分布普遍性的深层原因。

### 指数分布

$$p(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

描述事件的等待时间，具有无记忆性。

### Gamma 分布

$$p(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

指数分布是 $\alpha=1$ 的特例，卡方分布是特例。

---

## 3.3 条件概率与贝叶斯定理

### 条件概率

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**链式法则**（全概率公式的推广）：

$$P(A_1, A_2, \ldots, A_n) = P(A_1) \cdot P(A_2|A_1) \cdots P(A_n|A_1,\ldots,A_{n-1})$$

### 贝叶斯定理

$$\underline{\boldsymbol{P(\theta | X) = \frac{P(X | \theta) \cdot P(\theta)}{P(X)}}}$$

- $P(\theta)$：**先验**（Prior）—— 观测前对参数的信念
- $P(X|\theta)$：**似然**（Likelihood）—— 参数为 $\theta$ 时数据出现的概率
- $P(\theta|X)$：**后验**（Posterior）—— 看到数据后更新的信念
- $P(X)$：**边际似然**（归一化常数）

**直觉**：贝叶斯定理是"更新信念"的数学方法。

**示例（垃圾邮件）**：
- 先验： $P(\text{垃圾}) = 0.3$
- 似然： $P(\text{含"免费"}|\text{垃圾}) = 0.8$， $P(\text{含"免费"}|\text{正常}) = 0.1$
- 后验： $P(\text{垃圾}|\text{含"免费"}) = \frac{0.8 \times 0.3}{0.8 \times 0.3 + 0.1 \times 0.7} \approx 0.77$

---

## 3.4 信息论基础

### 信息熵（Shannon Entropy）

衡量随机变量的"不确定性"：

$$H(X) = -\sum_x P(X=x) \log P(X=x)$$

- 均匀分布时熵最大（最不确定）
- 确定性分布（$P=1$）时熵为 0

**示例**：公平硬币 $H = -0.5\log 0.5 - 0.5\log 0.5 = 1$ 比特

### 交叉熵

$$H(p, q) = -\sum_x p(x) \log q(x)$$

用分布 $q$ 来编码真实分布 $p$ 的信息的平均比特数。

**ML应用**：分类问题的损失函数：

$$L = -\sum_{i} y_i \log \hat{y}_i$$

其中 $y_i$ 是真实标签（one-hot）， $\hat{y}_i$ 是模型预测概率。

### KL 散度

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

衡量分布 $q$ 与真实分布 $p$ 的"差距"。

**性质**：
- $D_{\text{KL}}(p \| q) \geq 0$（Gibbs 不等式）
- $D_{\text{KL}}(p \| q) = 0 \Leftrightarrow p = q$
- **不对称**： $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$

**VAE 中的作用**：KL 散度项迫使编码器输出的分布 $q(\mathbf{z}|\mathbf{x})$ 接近标准正态先验 $p(\mathbf{z}) = \mathcal{N}(0, I)$：

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}[\log p(\mathbf{x}|\mathbf{z})]}_{\text{重建误差}} + \underbrace{D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{正则项}}$$

---

## 3.5 大数定律与中心极限定理

### 大数定律

若 $X_1, X_2, \ldots, X_n$ 是独立同分布（i.i.d.）随机变量，期望为 $\mu$：

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu \quad (n \to \infty)$$

**ML意义**：SGD 中，用小批量梯度估计全局梯度，大数定律保证其期望正确。

### 中心极限定理（CLT）

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

大样本情形下，样本均值近似服从正态分布（无论原始分布如何）。

---

## 3.6 极大似然估计（MLE）

### 定义

给定参数模型 $p(\mathbf{x};\theta)$ 和独立同分布数据 $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$，MLE 求：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n p(\mathbf{x}_i; \theta)$$

通常取对数（将连乘变求和）：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(\mathbf{x}_i; \theta)$$

### 逻辑回归的 MLE

$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}$

对数似然：

$$\ell(\mathbf{w}) = \sum_{i=1}^n \left[y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)\right]$$

最大化 $\ell$ 等价于最小化**负对数似然（Binary Cross-Entropy Loss）**：

$$L(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

梯度：

$$\nabla L = \frac{1}{n} X^T (\hat{\mathbf{y}} - \mathbf{y})$$

---

## 3.7 最大后验估计（MAP）

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta|\mathbf{X}) = \arg\max_\theta \left[\sum_i \log p(\mathbf{x}_i|\theta) + \log p(\theta)\right]$$

**MAP = MLE + 先验正则化**：

- 高斯先验 $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$ → 等价于 L2 正则化（Ridge）
- 拉普拉斯先验 $p(\theta_j) \propto e^{-\lambda|\theta_j|}$ → 等价于 L1 正则化（Lasso）

**直觉**：MAP 是 MLE 在引入了参数先验信念后的推广。数据越多，先验影响越小，MAP 趋向 MLE。

---

## 3.8 朴素贝叶斯分类器

**假设**：给定类别 $y$，特征 $x_1, \ldots, x_d$ 条件独立：

$$P(\mathbf{x}|y) = \prod_{j=1}^d P(x_j | y)$$

**分类决策**：

$$\hat{y} = \arg\max_y P(y) \prod_{j=1}^d P(x_j | y)$$

**垃圾邮件过滤**：
- 特征：词语是否出现
- 对每个词语，统计在垃圾/正常邮件中的出现频率
- 对新邮件，计算各类后验概率，取最大者

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = ["buy cheap meds now", "meeting at 3pm", "free money click here", "project update"]
labels = [1, 0, 1, 0]   # 1=垃圾, 0=正常

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

clf = MultinomialNB()
clf.fit(X, labels)

test = vectorizer.transform(["free meds click"])
print(clf.predict(test))  # [1] — 垃圾邮件
```

---

## 3.9 贝叶斯推断与变分自编码器

### 贝叶斯线性回归

先验： $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}I)$

似然： $y | \mathbf{x}, \mathbf{w} \sim \mathcal{N}(\mathbf{w}^T\mathbf{x}, \beta^{-1})$

后验（由于高斯共轭先验）：

$$p(\mathbf{w}|\mathbf{X}, \mathbf{y}) = \mathcal{N}(\mathbf{w}; \mathbf{m}_N, S_N)$$

$$S_N^{-1} = \alpha I + \beta X^T X, \quad \mathbf{m}_N = \beta S_N X^T \mathbf{y}$$

**优点**：输出预测分布（包含不确定性），而非点估计。

### VAE 中的 KL 散度

VAE 编码器输出 $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，先验 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$，两高斯间 KL 散度有闭合解：

$$D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

---

# 第四模块：图论

## 4.1 图的基本概念

### 图的定义

图 $G = (V, E)$：
- $V$：顶点集， $|V| = n$
- $E \subseteq V \times V$：边集， $|E| = m$

**分类**：

| 类型 | 描述 | 示例 |
|------|------|------|
| 无向图 | 边无方向 | 社交网络中的"朋友"关系 |
| 有向图（有向图） | 边有方向 | 网页链接、引用关系 |
| 加权图 | 边有权重 | 距离、相似度 |
| 无权图 | 边仅表示连接 | 0/1 邻接 |

### 邻接矩阵

$$A_{ij} = \begin{cases} 1 & (i,j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**无向图**的邻接矩阵对称： $A = A^T$。

**加权图**： $A_{ij} = w_{ij}$。

**示例**：三角形图（顶点 0, 1, 2，三条边）：

$$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

### 度矩阵

$$D_{ii} = \text{deg}(i) = \sum_j A_{ij}$$

$D$ 是对角矩阵， $D_{ii}$ 是顶点 $i$ 的度数（连接的边数）。

对上例： $D = \text{diag}(2, 2, 2)$。

### 图信号

图信号 $\mathbf{f}: V \to \mathbb{R}$，即每个顶点上有一个实数值，组成向量 $\mathbf{f} \in \mathbb{R}^n$。

**示例**：社交网络中，每个人的年龄、收入就是图信号。

---

## 4.2 拉普拉斯矩阵

### 定义

$$L = D - A$$

**示例**（三角形图）：

$$L = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix}$$

### 归一化拉普拉斯

$$\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

特征值在 $[0, 2]$ 之间，更适合谱分析。

### 拉普拉斯算子的意义

$$（L\mathbf{f})_i = \sum_{j \sim i} (f_i - f_j)$$

顶点 $i$ 的信号与其邻居信号之差的总和——衡量信号的"局部变化量"。

类比连续域的拉普拉斯算子 $\Delta f = \nabla^2 f$（描述函数的"曲率"）。

### 谱性质

**定理**：

1. $L$ 是实对称半正定矩阵（所有特征值 $\lambda_i \geq 0$）
2. 最小特征值 $\lambda_1 = 0$，对应特征向量 $\mathbf{1}/\sqrt{n}$（全1向量）
3. **零特征值的重数 = 图的连通分量数**
4. 第二小特征值 $\lambda_2 > 0 \Leftrightarrow$ 图连通

**推导 $L$ 半正定**：

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T D \mathbf{x} - \mathbf{x}^T A \mathbf{x} = \sum_i d_i x_i^2 - \sum_{(i,j)\in E} 2 x_i x_j = \sum_{(i,j)\in E} (x_i - x_j)^2 \geq 0$$

---

## 4.3 图傅里叶变换

### 图傅里叶基

将 $L$ 做特征分解： $L = U \Lambda U^T$，特征向量矩阵 $U = [\mathbf{u}_1, \ldots, \mathbf{u}_n]$。

这些特征向量构成图的**傅里叶基**（类比连续域的正弦/余弦函数）。

- 小特征值的特征向量：变化平缓（低频）
- 大特征值的特征向量：变化剧烈（高频）

### 图傅里叶变换

**正变换**（时域 → 频域）：

$$\hat{\mathbf{f}} = U^T \mathbf{f}$$

**逆变换**（频域 → 时域）：

$$\mathbf{f} = U \hat{\mathbf{f}}$$

### 图上的卷积

在图上，卷积在频域定义（利用卷积定理）：

$$\mathbf{f} * \mathbf{g} = U \left[ (U^T \mathbf{f}) \odot (U^T \mathbf{g}) \right]$$

对应的图滤波器（滤波器参数 $\theta$）：

$$\mathbf{y} = U \text{diag}(\theta) U^T \mathbf{f} = g_\theta(L) \mathbf{f}$$

**问题**：直接计算需要 $L$ 的特征分解， $O(n^3)$ 太贵。

---

## 4.4 切比雪夫多项式逼近

**思路**：用切比雪夫多项式逼近滤波器，避免特征分解。

**切比雪夫多项式**（递推定义）：

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$

**滤波器近似**：

$$g_\theta(L) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L})$$

其中 $\tilde{L} = \frac{2L}{\lambda_{\max}} - I$ 是归一化的拉普拉斯（将特征值范围缩放到 $[-1,1]$）。

**优点**：
- 无需特征分解，直接用矩阵-向量乘法
- $K$ 阶多项式意味着 $K$ 跳邻居的信息聚合

---

## 4.5 图卷积网络（GCN）

### Kipf & Welling (2017) 简化

取 $K=1$ 的切比雪夫近似，并令 $\lambda_{\max} \approx 2$：

$$g_\theta * \mathbf{f} \approx \theta_0 \mathbf{f} + \theta_1 (L - I) \mathbf{f} = \theta_0 \mathbf{f} - \theta_1 D^{-1/2} A D^{-1/2} \mathbf{f}$$

令 $\theta = \theta_0 = -\theta_1$，并加入自环 $\tilde{A} = A + I$：

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

其中：
- $\tilde{A} = A + I$（加自环使每个节点聚合自身信息）
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$（新度矩阵）
- $H^{(l)}$：第 $l$ 层节点特征矩阵
- $W^{(l)}$：可训练权重矩阵

### 消息传递视角

每个节点聚合自身和邻居的特征，做加权平均后通过线性变换和激活函数：

$$\mathbf{h}_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \frac{1}{|\mathcal{N}(i)|+1} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \mathbf{h}_j^{(l)}\right)$$

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## 4.6 图嵌入

### DeepWalk

思路：在图上做**随机游走**，生成节点序列，再用 Word2Vec 学习节点嵌入。

1. 从每个节点出发，做 $t$ 步随机游走（等概率选择邻居）
2. 得到大量节点序列（类比句子）
3. 用 Skip-gram 优化：最大化序列中共现节点的内积

$$\max \sum_{v \in V} \sum_{u \in \mathcal{N}_R(v)} \log P(u | \mathbf{z}_v)$$

### Node2Vec

在 DeepWalk 基础上，引入两个参数控制游走策略：

- **返回参数 $p$**：控制回到上一节点的概率
- **出入参数 $q$**：控制游走是局部（DFS-like）还是全局（BFS-like）

**直觉**：
- $p$ 小：趋向 BFS（探索同质性，同社区节点嵌入相近）
- $q$ 小：趋向 DFS（探索结构等价性，结构相似节点嵌入相近）

---

## 4.7 谱聚类

### 算法步骤

1. 构建图（如 k 近邻图）和拉普拉斯矩阵 $L$
2. 计算 $L$ 的前 $k$ 个最小特征向量 $U_k \in \mathbb{R}^{n \times k}$
3. 将每个节点表示为 $U_k$ 的对应行（$k$ 维向量）
4. 对这些向量做 K-Means 聚类

### 为何用第二小特征向量？

**Fiedler 向量**（$\lambda_2$ 对应的特征向量）：正值节点和负值节点正好对应图的两个部分。

谱聚类通过最小化图切割（Graph Cut）来分割图：连接两社区的边数少，即 $\sum_{(i,j)\in \text{cut}} (f_i - f_j)^2 = \mathbf{f}^T L \mathbf{f}$ 小。

```python
from sklearn.cluster import SpectralClustering
import numpy as np

# 邻接矩阵
A = np.array([[0,1,1,0,0],
              [1,0,1,0,0],
              [1,1,0,1,0],
              [0,0,1,0,1],
              [0,0,0,1,0]])

sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0)
labels = sc.fit_predict(A)
print(labels)  # 如 [0, 0, 0, 1, 1]
```

---

## 4.8 PageRank

### 问题

如何对网页按重要性排名？

### 随机游走解释

用户随机点击网页链接，最终会以某个**稳态概率**分布 $\boldsymbol{\pi}$ 停留在各页面。重要的页面 = 被重要页面链接多的页面（递归定义）。

### 数学形式

**转移矩阵** $M$： $M_{ij} = \frac{1}{d_j}$ 若存在边 $j \to i$，否则为 0。

**Power iteration**：

$$\mathbf{r}_{t+1} = \alpha M \mathbf{r}_t + \frac{1-\alpha}{n} \mathbf{1}$$

- $\alpha \approx 0.85$：阻尼因子（$1-\alpha$ 的概率随机跳到任意页面）
- 迭代直至收敛： $\mathbf{r}^* = $ 稳态向量

**收敛到**： $M$ 的最大特征向量（对应特征值 1）。

```python
def pagerank(A, alpha=0.85, max_iter=100):
    n = A.shape[0]
    # 按列归一化（转移矩阵）
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    M = A / col_sums
    r = np.ones(n) / n
    for _ in range(max_iter):
        r = alpha * M @ r + (1 - alpha) / n
    return r

A = np.array([[0,1,1],[1,0,0],[0,1,0]], dtype=float)
r = pagerank(A)
print("PageRank:", r)
```

---

## 4.9 最大流 / 最小割

### 最大流定理

**定理（Max-Flow Min-Cut）**：

$$\text{最大流} = \text{最小割容量}$$

**最小割**：将节点集分为 $S$（包含源点）和 $T$（包含汇点）两组，割去所有 $S \to T$ 的边，最小化这些边的容量之和。

**应用**：
- 网络流量最优化
- 图像分割（GraphCut）
- 社区发现

---

# 综合总结与学习路线

## 知识模块关系图

```
线性代数
  ├── 特征分解 ──→ PCA ──→ 降维、特征脸
  ├── SVD ──→ 推荐系统、低秩近似
  └── 正规方程 ──→ 最小二乘、Ridge/Lasso

微积分
  ├── 梯度 ──→ 梯度下降 ──→ SGD、Adam
  ├── 链式法则 ──→ 反向传播 ──→ 所有神经网络训练
  └── 拉格朗日/KKT ──→ SVM、约束优化

概率统计
  ├── 贝叶斯定理 ──→ MAP、朴素贝叶斯
  ├── MLE ──→ 逻辑回归损失、所有生成模型
  └── KL散度 / 交叉熵 ──→ 分类损失、VAE

图论
  ├── 拉普拉斯谱 ──→ 图傅里叶、谱聚类
  ├── GCN ──→ 节点分类、图分类
  └── PageRank ──→ 搜索引擎、节点重要性
```

## 优先学习顺序建议

1. **第一优先级**（必须掌握）
   - 矩阵乘法、转置、逆
   - 特征值/特征向量
   - 梯度与链式法则
   - 贝叶斯定理、MLE
   - 交叉熵损失

2. **第二优先级**（理解深度学习必备）
   - SVD 与低秩近似
   - 海森矩阵与二阶优化
   - 高斯分布的各种性质
   - KL 散度

3. **第三优先级**（进阶主题）
   - 图拉普拉斯谱理论
   - 切比雪夫近似与 GCN
   - 变分推断
   - 凸优化理论

## 代码工具推荐

```python
# 线性代数
import numpy as np
np.linalg.eig(), np.linalg.svd(), np.linalg.solve()

# 自动微分（反向传播）
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x**2).sum()
y.backward()  # 自动计算梯度
print(x.grad)  # tensor([2., 4.])

# 概率与统计
import scipy.stats as stats
stats.norm.pdf(), stats.multivariate_normal.pdf()

# 图
import networkx as nx
G = nx.karate_club_graph()
nx.pagerank(G, alpha=0.85)
```

---

> **学习建议**：每个概念要求能做到三件事：① 用自己的话解释定义；② 写出核心数学公式；③ 给出一个具体的ML应用例子。完成这三点，即为真正掌握。
