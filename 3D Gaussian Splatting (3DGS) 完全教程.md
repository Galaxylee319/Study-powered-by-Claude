# 3D Gaussian Splatting (3DGS) 完全教程

---

## 第一节：背景与基础概念

### 1.1 问题的起点：新视角合成

**新视角合成（Novel View Synthesis, NVS）** 是计算机视觉的核心问题之一：给定若干张从不同角度拍摄的照片，能否渲染出任意新视角下的图像？

这个问题的应用极广：VR/AR、影视特效、数字孪生、自动驾驶仿真。

### 1.2 3DGS之前的主流方法：NeRF

2020年，**NeRF（Neural Radiance Field）** 横空出世。它的思路是：

- 用一个 MLP 神经网络隐式地表示三维场景
- 网络输入：空间坐标 $(x,y,z)$ + 观察方向 $(\theta, \phi)$
- 网络输出：颜色 $(r,g,b)$ + 体密度 $\sigma$
- 渲染时沿射线做体积积分（Ray Marching）

NeRF的问题：**训练慢、推理更慢**（每帧需要对每条射线采样数百个点，每个点都要过神经网络）。

### 1.3 3DGS的核心思想

2023年，Kerbl et al. 在 SIGGRAPH 发表了 **3D Gaussian Splatting**，彻底解决了速度问题：

| 对比维度 | NeRF | 3DGS |
|---|---|---|
| 场景表示 | 隐式（神经网络） | **显式（三维高斯椭球）** |
| 渲染方式 | Ray Marching | **Splatting（投影光栅化）** |
| 训练时间 | 数小时～数天 | **约30分钟** |
| 实时渲染 | ❌ | **✅（>100 FPS）** |
| 可编辑性 | 差 | **好** |

核心直觉：**用一堆三维"高斯椭球"来表示场景，每个椭球有位置、形状、颜色和透明度，渲染时把它们"溅射"（splat）到二维图像平面上叠加合成。**

---

## 第二节：三维高斯的数学表示

### 2.1 什么是三维高斯

一个三维高斯函数由以下参数完整描述：

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

- $\boldsymbol{\mu} \in \mathbb{R}^3$：**均值（中心位置）**
- $\Sigma \in \mathbb{R}^{3\times3}$：**协方差矩阵（控制形状和朝向）**，必须是正定对称矩阵

### 2.2 协方差矩阵的参数化

直接优化 $\Sigma$ 会破坏正定性约束，因此将其分解：

$$\Sigma = R S S^T R^T$$

其中：
- $S = \text{diag}(s_x, s_y, s_z)$：**缩放矩阵**，$s_i > 0$ 控制三个轴向的大小
- $R$：**旋转矩阵**，由单位四元数 $\mathbf{q} = (q_w, q_x, q_y, q_z)$ 参数化

这样优化变量变为 $(s_x, s_y, s_z, q_w, q_x, q_y, q_z)$，天然保证 $\Sigma$ 正定。

**几何直觉**：每个高斯是一个三轴可任意拉伸、任意旋转的"椭球体"。

### 2.3 外观：球谐函数（Spherical Harmonics）

每个高斯的颜色不是固定的，而是与**观察方向**有关（模拟镜面反射等效果）。用球谐函数（SH）参数化：

$$c(\mathbf{d}) = \sum_{l=0}^{L}\sum_{m=-l}^{l} k_{lm} Y_l^m(\mathbf{d})$$

- $\mathbf{d}$：单位观察方向向量
- $Y_l^m$：球谐基函数（预计算的固定函数）
- $k_{lm}$：**可学习的球谐系数**
- $L$：阶数，通常取3阶，每个颜色通道有 $(L+1)^2 = 16$ 个系数

低阶（$l=0$）对应漫反射，高阶对应方向相关的镜面效果。

### 2.4 完整的单个高斯参数列表

| 参数 | 维度 | 意义 |
|---|---|---|
| $\boldsymbol{\mu}$ | 3 | 中心坐标 |
| $\mathbf{q}$（四元数） | 4 | 旋转 |
| $\mathbf{s}$（缩放） | 3 | 三轴尺度 |
| $\alpha$（不透明度） | 1 | logit空间存储 |
| 球谐系数 | $3\times16=48$ | 视角相关颜色 |

**每个高斯共59个参数。** 一个典型场景有 $10^5 \sim 10^6$ 个高斯。

---

## 第三节：渲染——从3D到2D的Splatting

### 3.1 将三维高斯投影到二维

给定相机参数（内参 $K$，外参 $[R_c | \mathbf{t}_c]$），将三维高斯投影到图像平面。

**第一步：变换到相机空间**

$$\boldsymbol{\mu}' = R_c \boldsymbol{\mu} + \mathbf{t}_c$$

**第二步：投影协方差矩阵（EWA Splatting）**

直接非线性投影不可微，用 **局部仿射近似（Jacobian线性化）**：

定义投影函数 $\varphi: \mathbb{R}^3 \to \mathbb{R}^2$（透视投影），在 $\boldsymbol{\mu}'$ 处的 Jacobian：

$$J = \frac{\partial \varphi}{\partial \mathbf{x}}\Bigg|_{\boldsymbol{\mu}'} = \begin{pmatrix} \frac{f_x}{z'} & 0 & -\frac{f_x x'}{z'^2} \\ 0 & \frac{f_y}{z'} & -\frac{f_y y'}{z'^2} \end{pmatrix}$$

其中 $(x', y', z') = \boldsymbol{\mu}'$，$f_x, f_y$ 为焦距。

**二维投影协方差矩阵**：

$$\Sigma^{2D} = J R_c \Sigma R_c^T J^T$$

（取前 $2\times2$ 子矩阵）

这样，每个三维高斯在图像平面上对应一个**二维高斯椭圆**。

### 3.2 Alpha合成渲染

对图像上的每个像素 $\mathbf{p}$，找到所有覆盖该像素的高斯（按深度排序），做 **前到后（front-to-back）** 的 Alpha 合成：

**第 $i$ 个高斯对像素的贡献**：

$$\alpha_i' = \alpha_i \cdot \exp\left(-\frac{1}{2} \Delta_i^T (\Sigma_i^{2D})^{-1} \Delta_i\right)$$

其中 $\Delta_i = \mathbf{p} - \boldsymbol{\mu}_i^{2D}$ 是像素中心到高斯中心的二维偏差。

**最终像素颜色**（体渲染方程的离散化）：

$$C = \sum_{i=1}^{N} c_i \alpha_i' \prod_{j=1}^{i-1}(1 - \alpha_j')$$

这与 NeRF 的体渲染公式本质一致，但计算的是**离散高斯的叠加**而不是连续积分。

**透射率**（transmittance）：

$$T_i = \prod_{j=1}^{i-1}(1-\alpha_j')$$

当 $T_i < \epsilon$ 时可以提前终止（early stopping）。

### 3.3 Tile-based 光栅化（关键工程优化）

朴素地对每像素遍历所有高斯是 $O(N \times H \times W)$，不可接受。3DGS 的解决方案：

1. **Tile划分**：将图像划分为 $16\times16$ 的像素块（tile）
2. **Gaussian-Tile关联**：为每个高斯计算它覆盖哪些 tiles
3. **GPU排序**：用 CUDA 对每个 tile 内的高斯按深度做 Radix Sort
4. **并行渲染**：每个 tile 独立并行处理

这将复杂度降为 $O(\bar{k} \cdot H \times W)$，$\bar{k}$ 为每像素平均高斯数（通常远小于 $N$）。

---

## 第四节：训练——自适应密度控制

### 4.1 损失函数

给定训练图像 $I_{gt}$，渲染图像 $\hat{I}$，损失为：

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}$$

- $\mathcal{L}_1 = \|\hat{I} - I_{gt}\|_1$：像素级 L1 损失
- $\mathcal{L}_{D-SSIM}$：结构相似度损失，感知质量更好
- $\lambda = 0.2$（原论文默认值）

所有参数（位置、旋转、缩放、不透明度、球谐系数）全部通过梯度下降优化。

### 4.2 自适应密度控制（Adaptive Density Control）—— 3DGS 的精髓

初始化后高斯数量固定会导致欠拟合（覆盖不足）或过拟合（冗余）。3DGS 设计了动态调整机制：

#### 4.2.1 克隆与分裂（Densification）

**触发条件**：位置梯度均值 $\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\|$ 超过阈值 $\tau_{grad}$（说明该区域重建不足）

- **小高斯（欠重建区域）→ 克隆（Clone）**：
  在原高斯附近复制一个新高斯，略微随机偏移位置

- **大高斯（过重建区域）→ 分裂（Split）**：
  将原高斯分裂为两个更小的高斯，新尺度约为原来的 $1/\phi$（$\phi=1.6$）

```
每 100 次迭代检查一次：
  对每个高斯，计算平均位置梯度
  if 梯度 > τ_grad:
    if 高斯尺寸 < τ_size:  → Clone
    else:                  → Split
```

#### 4.2.2 剪枝（Pruning）

移除以下高斯：
- **不透明度过低**：$\alpha < \epsilon_\alpha$（几乎透明，无贡献）
- **过大的高斯**（世界空间或相机空间中过大，通常是floater伪影）

每 $N_{reset}=3000$ 次迭代，将所有 $\alpha$ 重置为接近0（强制重新竞争），再通过梯度学习决定哪些应保留。

#### 4.2.3 完整训练流程

```
初始化：从SfM点云（COLMAP）得到稀疏点，每点创建一个高斯

for iter = 1 to 30000:
    随机采样一张训练图像
    渲染图像（Tile-based Splatting）
    计算损失 L
    反向传播，更新所有参数
    
    if iter % 100 == 0 and iter < 15000:
        自适应密度控制（Clone/Split/Prune）
    
    if iter % 3000 == 0:
        重置不透明度
```

### 4.3 梯度流分析

关键在于渲染管线全程可微（除排序操作外）：

- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}^{2D}}$：驱动位置移动到更好的地方
- $\frac{\partial \mathcal{L}}{\partial \Sigma^{2D}}$：驱动形状/朝向调整
- 链式法则反传回 $\frac{\partial \mathcal{L}}{\partial \mathbf{q}}, \frac{\partial \mathcal{L}}{\partial \mathbf{s}}$

排序操作（Radix Sort）不可微，但实践中**将排序视为常数**（即本次渲染用的排序结果不参与梯度计算），效果依然很好，因为排序只影响合成顺序，高斯整体优化目标不变。

---

## 第五节：初始化——COLMAP与SfM

3DGS 不是从零初始化，而是依赖**运动恢复结构（SfM）**：

### 5.1 流程

```
输入图像 → COLMAP SfM → 稀疏点云 + 相机位姿
                ↓
         3DGS 初始化每个点为一个高斯
                ↓
         训练优化（自适应密度控制）
```

### 5.2 初始高斯参数设置

- **位置** $\boldsymbol{\mu}$：SfM 点的三维坐标
- **缩放** $\mathbf{s}$：根据最近邻距离估计：$s = \text{mean}(\|\boldsymbol{\mu} - \text{kNN}\|)$
- **旋转** $\mathbf{q}$：单位四元数（球形初始化）
- **不透明度** $\alpha$：0.1（logit空间存储）
- **球谐系数**：零阶项由SfM点颜色初始化，高阶项为0

---

## 第六节：与NeRF的深度对比

### 体渲染方程的统一视角

NeRF 渲染方程（连续形式）：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

$$T(t) = \exp\!\left(-\int_{t_n}^t \sigma(\mathbf{r}(s))\,ds\right)$$

3DGS 渲染方程（离散形式）：

$$C = \sum_i T_i \alpha_i' c_i, \quad T_i = \prod_{j<i}(1-\alpha_j')$$

两者本质上是同一体渲染框架的**连续 vs 离散**实例。3DGS 把连续密度场替换成了有限个高斯椭球，因此可以做高效的光栅化而不是慢速的射线步进。

---

## 第七节：经典开源项目与应用

### 7.1 原始论文代码

**gaussian-splatting**（Kerbl et al., SIGGRAPH 2023）
- 📦 https://github.com/graphdeco-inria/gaussian-splatting
- CUDA + PyTorch 实现，包含完整训练和实时查看器
- 依赖：COLMAP、custom CUDA rasterizer

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
# 安装SIBR viewer用于实时渲染
```

### 7.2 重要衍生工作

| 项目 | 改进点 | 地址 |
|---|---|---|
| **Mip-Splatting** | 抗锯齿，解决缩放伪影 | github.com/autonomousvision/mip-splatting |
| **2DGS** | 用二维高斯圆盘，几何更精确 | github.com/hbb1/2d-gaussian-splatting |
| **4DGS** | 加时间维度，动态场景重建 | github.com/hustvl/4DGaussians |
| **GaussianAvatars** | 人体/头像重建 | github.com/ShenhanQian/GaussianAvatars |
| **Gaussian Opacity Fields** | 精确表面重建与Mesh提取 | github.com/autonomousvision/gaussian-opacity-fields |
| **LangSplat** | 3D语义理解（CLIP特征蒸馏） | github.com/minghanqin/LangSplat |
| **GaussianEditor** | 文字指令编辑3DGS场景 | github.com/buaacyw/GaussianEditor |
| **gsplat** | 高性能Python库，适合二次开发 | github.com/nerfstudio-project/gsplat |

### 7.3 实用工具链

**COLMAP**（相机位姿估计）：
```bash
colmap automatic_reconstructor --workspace_path ./data --image_path ./data/images
```

**gsplat**（最易用的库）：
```python
from gsplat import rasterization

# 渲染一帧
renders, alphas, info = rasterization(
    means=gaussians.means,      # [N, 3]
    quats=gaussians.quats,      # [N, 4]
    scales=gaussians.scales,    # [N, 3]
    opacities=gaussians.opacities,  # [N]
    colors=gaussians.colors,    # [N, C]
    viewmats=viewmat[None],     # [1, 4, 4]
    Ks=K[None],                 # [1, 3, 3]
    width=W, height=H,
)
```

**nerfstudio**（统一框架，支持多种方法）：
```bash
pip install nerfstudio
ns-train splatfacto --data ./data
```

### 7.4 查看器与格式

训练完成后得到 `point_cloud.ply` 文件（存储所有高斯参数），可用：

- **SuperSplat**（网页端）：https://playcanvas.com/supersplat/editor — 直接浏览器拖入查看
- **3DGS Viewer**（桌面端）：https://github.com/mkkellogg/GaussianSplats3D（Three.js）
- **Luma AI**：商业级移动端渲染

---

## 第八节：局限性与前沿方向

### 8.1 当前局限

- **内存占用大**：百万高斯 × 59参数，大场景可达数GB
- **动态场景**：原始3DGS不支持运动物体
- **精确几何**：高斯是体积基元，提取精确Mesh较困难
- **室外大场景**：COLMAP在大场景下不稳定，无界场景需特殊处理

### 8.2 前沿研究方向

- **压缩**：Scaffold-GS、Compact3DGS 等用锚点+神经网络压缩参数
- **大场景**：VastGaussian、CityGaussian 分块处理城市级场景
- **物理仿真**：PhysGaussian 让高斯遵从物理形变
- **生成模型**：GaussianDreamer 结合扩散模型做3DGS生成
- **SLAM**：Gaussian Splatting SLAM（你的GSoC方向与此高度相关！）将3DGS用作在线建图表示

---

## 总结：3DGS知识图谱

```
输入：多视角图像
    ↓
COLMAP SfM → 稀疏点云 + 相机位姿
    ↓
初始化：每点 → 一个三维高斯 (μ, Σ=RSS^TR^T, α, SH系数)
    ↓
训练循环（30k次迭代）：
  ├─ 前向：EWA投影 → Tile排序 → Alpha合成 → 渲染图像
  ├─ 损失：L1 + D-SSIM
  ├─ 反向：梯度传播到所有59个参数
  └─ 每100次：自适应密度控制（Clone/Split/Prune）
    ↓
输出：point_cloud.ply（高斯场）→ 实时渲染 >100FPS
```

---
