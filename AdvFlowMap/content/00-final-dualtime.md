# 最终版：从 AdvFM 到 Dual-Time Video Adversarial Manifold

> **核心定位：AdvFM 是可行性起点，不是我们的方法终点。** AdvFM 已在图像域表明：Flow-Matching velocity-field attack 可借助 Gaussian smoothing 降低梯度方差、提高 surrogate–target alignment，并使扰动更集中于 robust-tangent directions；lookahead 进一步降低沿 ODE trajectory 的 temporal mismatch。我们的核心问题是：**视频多出真实的 Physical Video-Time，真正可迁移的视频对抗方向是否必须同时在 Generative Flow-Time 与 Physical Video-Time 上保持稳定？**

## 最终题目

# **AdvFlowMap**
## **Discovering Transferable Adversarial Video Manifolds via Dual-Time Flow Transport**

一句话主线：

> **Flow is not merely an attack space; it is a manifold explorer. Transferable video adversarial directions should be jointly stable under generative flow transport and semantic temporal transport.**

---

# 一、总：最终 CVPR Research 总框图

```text
STARTING POINT — AdvFM (Image)
        │
        ├─ Gaussian / Flow smoothing → gradient variance ↓
        ├─ velocity-field transport → adversarial accumulation
        ├─ robust-tangent tendency → shared/robust alignment ↑
        └─ lookahead → ODE temporal mismatch ↓
        │
        ▼
Flow trajectory is a useful carrier of transferable directions
        │
        │ BUT: image has only one generative time
        ▼
VIDEO = TWO TIMES
        │
        ├─ s : Generative Flow-Time
        └─ τ : Physical Video-Time
        │
        ▼
CORE SCIENTIFIC QUESTION
“什么样的方向能跨异构视频模型稳定迁移？”
        │
        ▼
DUAL-TIME TRANSPORT INVARIANCE
        │
        ├─ Flow-Time Invariance
        ├─ Temporal Semantic Equivariance
        └─ Flow↔Time Commutative Consistency
        │
        ▼
ADVERSARIAL TRANSPORT HOLONOMY
H_adv ↓
        │
        ▼
Model/Frame-specific Components ↓
Shared Adversarial Ratio ↑
        │
        ▼
Target Gradient Alignment ↑
        │
        ▼
Black-box Transfer ASR ↑
        │
        ▼
METHOD — AdvFlowMap
        ├─ MFTP: Multi-Interval Flow Transport Probing
        ├─ DTIM: Dual-Time Invariant Direction Mining
        └─ DLA : Dual-Lookahead Adversarial Optimization
        │
        ▼
VideoMAE / VideoSwin / TimeSformer / SlowFast
Cross-Architecture Transfer ↑
```

---

# 二、现象：AdvFM 给我们的三个 Transfer Principles

## Principle A — Smoothing → Shared Gradient

设 surrogate gradient 为：

\[
g_s=g_*+\epsilon_s,
\]

其中 \(g_*\) 表示跨模型共享 vulnerability，\(\epsilon_s\) 是 surrogate-specific component。AdvFM 的核心启示之一是：Gaussian / flow smoothing 降低更新方向方差，因此更容易提高 surrogate–target gradient alignment。

我们的升级不是再次证明 smoothing 有效，而是问：

> **finite-time natural transport 能否进一步过滤 model-specific component，并暴露稳定的 shared adversarial direction？**

## Principle B — Manifold Tangent → Robust Transfer

AdvFM 提示 transferable perturbation 更倾向于 robust/manifold-tangent directions。由此得到我们的几何假设：

\[
\boxed{\text{Transferable Direction}\approx\text{Stable Manifold-Tangent Direction}}
\]

## Principle C — Lookahead → Transport Consistency

AdvFM 的 lookahead 不只优化当前 flow state，而同时考虑 rolled-out state，减少 generative ODE trajectory 上的 mismatch。

但是视频还存在第二种 mismatch：

\[
\boxed{\text{Generative-Time Mismatch}+\text{Physical-Video-Time Mismatch}}
\]

这正是 AdvFlowMap 的科学入口。

---

# 三、核心科学问题：视频不是一条轨迹，而是一张 Dual-Time Surface

视频表示为：

\[
X=[x^1,x^2,\ldots,x^T].
\]

引入 generative flow-time \(s\) 后，状态变为：

\[
\boxed{X(s,\tau)}
\]

其中：

- \(s\)：Generative Flow-Time；
- \(\tau\)：Physical Video-Time。

因此视频形成二维 transport surface：

\[
\mathcal S_{video}=\{X(s,\tau)\}.
\]

存在两个 transport operators：

\[
\Phi_{s\to s'}:X(s,\tau)\to X(s',\tau),
\]

\[
\Psi_{\tau\to\tau'}:X(s,\tau)\to X(s,\tau').
\]

**真正的问题不是“怎样攻击 Flow Matching”，而是：哪些 adversarial vector fields 能同时沿 \(\Phi\) 和 \(\Psi\) 稳定传播？**

---

# 四、机制 1：Flow-Time Transport Invariance

从 instantaneous velocity 改为 finite-time transport：

\[
X_k=\Phi_{s_k\to r_k}(X).
\]

第 \(k\) 个 transport interval 对应：

\[
g_k=J_{\Phi_k}^{T}\nabla L(f_s(\Phi_k(X)),y).
\]

定义 Flow-Time Invariance：

\[
\boxed{
I_F(d)=\frac1K\sum_k
\cos\left(d(\Phi_k(X)),J_{\Phi_k}(X)d(X)\right)
}
\]

它衡量：一个攻击方向经过自然 Flow Transport 后，是否仍表示“同一个 adversarial semantic direction”。

**高 \(I_F\)**：transport-stable，可能更接近 shared component。  
**低 \(I_F\)**：transport-sensitive，更可能是 surrogate-local noise。

---

# 五、机制 2：Video-Time Semantic Transport Equivariance

单纯要求相邻帧梯度相似并不够，因为动作目标在像素坐标中持续运动。

从 frozen video/self-supervised encoder 得到 frame/token representation \(F^\tau\)，建立 semantic correspondence：

\[
A_{\tau\to\tau+1}=
\operatorname{Softmax}\left(\frac{F^\tau(F^{\tau+1})^T}{\sqrt D}\right).
\]

要求 adversarial direction 满足：

\[
\boxed{d^{\tau+1}\approx A_{\tau\to\tau+1}^{T}d^\tau}
\]

而不是简单的 \(d^{\tau+1}\approx d^\tau\)。

这意味着攻击信号跟随 hand / object / action token 的语义运动，而不是固定在 screen coordinate 上。

定义：

\[
I_T(d)=\frac1{T-1}\sum_\tau
\cos\left(d^{\tau+1},A_{\tau\to\tau+1}^{T}d^\tau\right).
\]

---

# 六、核心新发现：Dual-Time Commutativity

对于真正 intrinsic 的 adversarial direction，从 \((s,\tau)\) 到 \((s',\tau')\) 有两条路径：

```text
                 Generative Flow-Time s

      X(s,τ)  ───────────────►  X(s',τ)
        │                         │
        │                         │
 Video  │                         │ Video
 Time   │                         │ Time
        │                         │
        ▼                         ▼
    X(s,τ+1) ───────────────► X(s',τ+1)

                 Flow-Time

Path A = Flow → Temporal
Path B = Temporal → Flow
```

我们提出核心假设：

\[
\boxed{
\Psi_{\tau\to\tau'}\circ\Phi_{s\to s'}(d)
\approx
\Phi_{s\to s'}\circ\Psi_{\tau\to\tau'}(d)
}
\]

即 transferable direction 应具有较强的 **Dual-Time Commutative Consistency**。

---

# 七、核心指标：Adversarial Transport Holonomy

令：

\[
d_A=\Psi_{\tau\to\tau'}[J_{\Phi_{s\to s'}}d],
\]

\[
d_B=J_{\Phi_{s\to s'}}[\Psi_{\tau\to\tau'}(d)].
\]

定义：

\[
\boxed{
H_{adv}(d)=1-\cos(d_A,d_B)
}
\]

## 解释

- \(H_{adv}\approx0\)：先 Flow 后 Time 与先 Time 后 Flow 得到近似一致方向；
- \(H_{adv}\) 大：方向依赖某个 flow state / frame / surrogate local geometry。

因此整篇论文最关键的可验证规律是：

\[
\boxed{
H_{adv}\downarrow
\Rightarrow
Shared\ Ratio\uparrow
\Rightarrow
Target\ Gradient\ Alignment\uparrow
\Rightarrow
Transfer\ ASR\uparrow
}
\]

> **如果这条 empirical law 在多个模型/数据集上稳定成立，论文就从“新攻击方法”升级成“视频对抗迁移性的几何机制发现”。**

---

# 八、Transferable Adversarial Video Manifold

定义：

\[
\boxed{
\mathcal M_{TAV}=
\left\{X+\delta:\|\delta\|_p\le\epsilon,
I_F(\delta)\ge\gamma_F,
I_T(\delta)\ge\gamma_T,
H_{adv}(\delta)\le\gamma_H
\right\}
}
\]

它由三个性质共同限定：

1. **Flow Stability**；
2. **Temporal Transportability**；
3. **Low Dual-Time Holonomy**。

论文中“manifold”需要谨慎使用：先通过上述局部 transport geometry 建立定义与实证，再讨论其 manifold interpretation，避免 marketing-only claim。

---

# 九、方法：AdvFlowMap 只保留三个统一模块

## 1. MFTP — Multi-Interval Flow Transport Probing

用多个 finite-time Flow Maps 探索 natural video manifold，而不是只攻击单个 instantaneous velocity：

\[
\{\Phi_k(X)\}_{k=1}^{K}.
\]

作用：产生多个 natural transport views，暴露 transport-stable 与 transport-unstable adversarial components。

## 2. DTIM — Dual-Time Invariant Direction Mining

统一优化：

\[
\mathcal L_{DTI}
=
\lambda_F\mathcal L_F+
\lambda_T\mathcal L_T+
\lambda_H\mathcal L_H.
\]

三项不是三个独立 tricks，而是同一个目标的三个几何约束：

> **寻找 Dual-Time Transport-Invariant adversarial direction。**

## 3. DLA — Dual-Lookahead Adversarial Optimization

AdvFM 只沿 generative flow-time look ahead。视频中同时 look ahead：

\[
(s+\Delta s,\tau),\quad
(s,\tau+\Delta\tau),\quad
(s+\Delta s,\tau+\Delta\tau).
\]

定义：

\[
\mathcal L_{DLA}=
\lambda_0L(X_{s,\tau})+
\lambda_fL(X_{s+\Delta s,\tau})+
\lambda_tL(X_{s,\tau+\Delta\tau})+
\lambda_{ft}L(X_{s+\Delta s,\tau+\Delta\tau}).
\]

一句话区别：

> **AdvFM looks ahead along one generative trajectory; AdvFlowMap looks ahead over a two-dimensional video transport surface.**

最终：

\[
\mathcal L_{attack}
=
\mathcal L_{adv}
+\lambda_L\mathcal L_{DLA}
-\lambda_D\mathcal L_{DTI},
\]

\[
X_{n+1}^{adv}=\Pi_\epsilon\left[X_n^{adv}+\alpha\operatorname{sign}(d^*)\right].
\]

---

# 十、理论闭环：建议证明的三个命题

## Proposition 1 — Flow transport suppresses model-specific variance

设：

\[
g_s=g_*+\epsilon_s,
\qquad g_t=g_*+\epsilon_t.
\]

若 finite-time Flow operator \(P_F\) 对 unstable component 满足：

\[
\|P_F\epsilon_s\|\le\rho_F\|\epsilon_s\|,\qquad \rho_F<1,
\]

则应得到更低的 model-specific variance，并提高期望 target alignment。

## Proposition 2 — Semantic temporal transport suppresses frame-specific noise

若 shared action component 满足：

\[
g_*^{\tau+1}\approx A^Tg_*^\tau,
\]

而 frame-specific noise 不满足该 transport law，则 temporal-equivariant aggregation 应进一步降低噪声：

\[
\boxed{
Var(g_{dual})<Var(g_{flow})<Var(g_{pixel})
}
\]

并预测：

\[
Alignment_{dual}>Alignment_{flow}>Alignment_{pixel}.
\]

## Proposition 3 — Low holonomy selects shared directions

若 shared component 对两种 transport 近似 commute，而 model-specific component 不满足，则：

\[
H(g_*)<H(\epsilon_s).
\]

因此最小化 holonomy 应提高 shared component ratio，并最终提高 cross-model alignment。

> 注意：以上是需要证明和实验验证的理论命题，不应在实验完成前写成无条件定理。

---

# 十一、Mechanism Figure：比 ASR 柱状图更重要

建议 Figure 2 直接验证四个关系：

| Panel | X-axis | Y-axis | 预期 |
|---|---|---|---|
| (a) | Flow Invariance \(I_F\) | Target Gradient Alignment | 正相关 |
| (b) | Temporal Transport \(I_T\) | Target Gradient Alignment | 正相关 |
| (c) | Holonomy \(H_{adv}\) | Target Gradient Alignment | **负相关** |
| (d) | Holonomy \(H_{adv}\) | Black-box Transfer ASR | **负相关** |

核心 empirical law：

\[
\boxed{\text{Low Dual-Time Transport Holonomy}\Longleftrightarrow\text{High Transferability}}
\]

---

# 十二、必须做：AdvFM → Video 的递进消融

| Method | Flow Attack | Flow Lookahead | Finite-Time Map | Semantic Time Transport | Dual-Time Holonomy | Transfer ASR |
|---|---:|---:|---:|---:|---:|---:|
| Pixel MI-FGSM | ✗ | ✗ | ✗ | ✗ | ✗ | Base |
| Frame-wise AdvFM | ✓ | ✓ | ✗ | ✗ | ✗ | ↑ |
| Video AdvFM + naive TC | ✓ | ✓ | ✗ | simple | ✗ | ↑ |
| + Flow Map | ✓ | ✓ | ✓ | ✗ | ✗ | ↑↑ |
| + Semantic Transport | ✓ | ✓ | ✓ | ✓ | ✗ | ↑↑ |
| **AdvFlowMap** | ✓ | ✓ | ✓ | ✓ | ✓ | **↑↑↑** |

这张表直接回答 reviewer：**为什么不是 simply applying AdvFM frame-by-frame？**

---

# 十三、Prior-art 边界必须写清楚

## AdvFM — CVPR 2026

它拥有：image-domain velocity-field attack、Gaussian smoothing / variance–alignment explanation、robust-tangent interpretation、single-generative-time lookahead。

**我们不重复这些贡献，而把它们作为 feasibility evidence。**

## BMTC — IJCAI 2025

已经利用 temporal gradient consistency 与 background mixup 提升 action-recognition adversarial transfer。因此不能把“temporal consistency”本身写成 novelty。

我们的区别是：

\[
\text{Temporal Gradient Similarity}
\rightarrow
\text{Semantic Transport Equivariance}
\rightarrow
\text{Dual-Time Commutativity}.
\]

## FeatureFool — CVPR 2026

已证明 feature-map based zero-query video attacks 具有较强 transferability。因此我们的 novelty 不应是“feature transfer”，而是 **dual-time transport geometry**。

## FMVP — 2026

FMVP 是 Flow Matching video purification defense。它适合作为 defense-aware stress test，而不是方法主线。

---

# 十四、实验体系

## Datasets

\[
\boxed{UCF101+HMDB51+Kinetics400}
\]

UCF/HMDB：快速、大量 cross-model mechanism analysis。  
K400：主结果和 scale validation。

## Models

\[
\boxed{SlowFast,\ VideoSwin,\ TimeSformer,\ VideoMAE}
\]

最好再加入一个现代 video foundation encoder。

## 主指标

- Average Black-box Transfer ASR；
- target-gradient alignment（仅 mechanism analysis，攻击时不可访问 target）；
- SSIM / PSNR / LPIPS；
- temporal perceptual consistency；
- runtime / NFE；
- purification / compression / resize / frame-sampling robustness。

---

# 十五、最关键的 Go / No-Go 实验

先不要训练完整系统。先在 VideoMAE surrogate 上做小规模验证，target 用 SlowFast / VideoSwin / TimeSformer。

比较：

\[
g_{pixel},\quad g_{AdvFM},\quad g_{flow},\quad g_{dual}.
\]

如果稳定得到：

\[
\boxed{
Align(g_{dual})>Align(g_{flow})>Align(g_{AdvFM})>Align(g_{pixel})
}
\]

并且：

\[
\boxed{
H_{adv}\downarrow\Longleftrightarrow TargetAlignment\uparrow,\ TransferASR\uparrow
}
\]

则核心科学假设成立，可以全力推进。

如果 holonomy 与 transferability 无稳定关系，则不要硬讲 holonomy；退回 **Flow-Invariant Direction + Semantic Temporal Transport**，仍可形成完整攻击论文。

---

# 十六、最终 Contributions：只写三条

### 1. Scientific Discovery

我们提出并验证：transferable video adversarial directions 具有 **dual-time transport invariance**；尤其是低 adversarial transport holonomy 与更高的 target-gradient alignment / black-box transferability 相关。

### 2. Mechanism & Theory

我们把视频对抗迁移建模在 \(X(s,\tau)\) 的 two-time transport surface 上，以 variance → invariance → alignment 的视角解释 Flow smoothing、semantic temporal transport 与 dual-time commutativity 如何逐级抑制 model/frame-specific components。

### 3. Method

提出 **AdvFlowMap**：Multi-Interval Flow Transport Probing + Dual-Time Invariant Direction Mining + Dual-Lookahead Optimization，在异构视频识别 architecture 之间生成高迁移且时序一致的 adversarial videos。

---

# 十七、总：最终 Research Loop

```text
AdvFM feasibility evidence
        ↓
Flow smoothing / tangent / lookahead
        ↓
Video introduces a second time axis
        ↓
X(s, τ): Dual-Time Video Surface
        ↓
Flow-Time Invariance
        +
Semantic Temporal Equivariance
        ↓
Flow ↔ Time Commutativity
        ↓
Adversarial Transport Holonomy
        ↓
Low H_adv selects shared directions
        ↓
Target Gradient Alignment ↑
        ↓
AdvFlowMap
        ↓
Cross-Architecture Video Transfer ↑
```

> ## 最终核心结论
> **视频对抗迁移性不应只理解为“找到更强的 surrogate gradient”。更有解释力的假设是：跨模型共享的 adversarial directions 是自然视频 manifold 上对两种时间 transport 都稳定的内禀方向。AdvFM 证明了 generative-flow transport 的可行性；AdvFlowMap 则进一步探索 generative flow-time × physical video-time 的双时间几何，并用低 transport holonomy 去发现真正 transferable 的视频对抗方向。**
