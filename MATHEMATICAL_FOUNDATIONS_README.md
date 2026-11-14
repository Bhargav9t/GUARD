# GUARD: Mathematical Foundations & Theoretical Formulations

This document provides a comprehensive, rigorous mathematical specification of all algorithms, signal processing methods, probabilistic models, and neural architectures implemented within **GUARD (Generalizable Universal AI-generated & Real-image Detector)**.

---

## Table of Contents
1. [2D Frequency Domain & Spectral Physics](#1-2d-frequency-domain--spectral-physics)
2. [Natural Image Statistics & Power-Law Decay](#2-natural-image-statistics--power-law-decay)
3. [Stochastic Camera Sensor Noise Residual Modeling](#3-stochastic-camera-sensor-noise-residual-modeling)
4. [Deep Neural Spatial Stream & Squeeze-and-Excitation Attention](#4-deep-neural-spatial-stream--squeeze-and-excitation-attention)
5. [Swin Transformer Shifted Window Attention](#5-swin-transformer-shifted-window-attention)
6. [Dynamic Gated Attention Cross-Modal Fusion](#6-dynamic-gated-attention-cross-modal-fusion)
7. [Gradient-Weighted Class Activation Mapping (Grad-CAM)](#7-gradient-weighted-class-activation-mapping-grad-cam)
8. [Temporal Video Forensics & Scene-Cut Filtering](#8-temporal-video-forensics--scene-cut-filtering)
9. [Optimization, Loss Functions & Regularization](#9-optimization-loss-functions--regularization)
10. [Decision Theory, Threshold Calibration & Forensic Metrics](#10-decision-theory-threshold-calibration--forensic-metrics)

---

## 1. 2D Frequency Domain & Spectral Physics

Generative models (GANs, Latent Diffusion Models, VAEs) synthesize images through sequences of spatial upsampling operations (transposed convolutions, nearest-neighbor interpolation, sub-pixel pixel-shuffling). These leave high-frequency grid artifacts and spectral roll-off anomalies invisible in pixel space.

### 1.1 2D Orthonormal Discrete Fourier Transform (2D DFT)
Given a discrete 2D image plane $f(x, y) \in \mathbb{R}^{H \times W}$, its continuous complex frequency representation $F(u, v)$ is defined as:

$$F(u, v) = \frac{1}{\sqrt{HW}} \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} f(x, y) \exp\left[-i 2\pi \left(\frac{ux}{H} + \frac{vy}{W}\right)\right]$$

where $u \in \{0, 1, \dots, H-1\}$ and $v \in \{0, 1, \dots, W-1\}$ index discrete spatial frequencies in cycles per dimension. Using Euler's formula: 
$$\exp(-i\theta) = \cos(\theta) - i\sin(\theta)$$

The complex spectrum decomposes into real and imaginary parts:

$$F(u, v) = \text{Re}(u, v) + i \cdot \text{Im}(u, v)$$

### 1.2 Quadrant Centering & Logarithmic Dynamic Range Compression
The zero-frequency Direct Current (DC) component $F(0, 0)$ is shifted to the geometric center $(u_0, v_0) = \left(\lfloor H/2 \rfloor, \lfloor W/2 \rfloor\right)$ using `fftshift`:

$$F_{\text{shifted}}(u, v) = F\left(\left(u - \frac{H}{2}\right) \bmod H, \left(v - \frac{W}{2}\right) \bmod W\right)$$

Because frequency magnitudes vary over orders of magnitude ($10^0$ to $10^7$), we apply logarithmic compression:

$$\mathbf{S}_{\text{FFT}}(u, v) = \log\left(1 + \sqrt{\text{Re}^2(u, v) + \text{Im}^2(u, v)}\right)$$

### 1.3 2D Discrete Cosine Transform (DCT-II)
To detect cardinal JPEG 8×8 block boundary anomalies, GUARD employs the orthonormal 2D DCT-II:

$$X(k_1, k_2) = \alpha(k_1)\alpha(k_2) \sum_{n_1=0}^{H-1} \sum_{n_2=0}^{W-1} f(n_1, n_2) \cos\left[\frac{\pi(2n_1 + 1)k_1}{2H}\right] \cos\left[\frac{\pi(2n_2 + 1)k_2}{2W}\right]$$

where:

$$\alpha(k) = \begin{cases} \sqrt{\frac{1}{N}}, & k = 0 \\ \sqrt{\frac{2}{N}}, & k > 0 \end{cases}$$

---

## 2. Natural Image Statistics & Power-Law Decay

### 2.1 Scale-Invariance and Power-Law Falloff
In natural optical images captured by physical lens-sensor systems, spatial structures exhibit scale invariance (Field, 1987; Torralba & Oliva, 2003). The expected radially averaged power spectral density $P(f)$ follows a power law:

$$P(f) \approx \frac{A}{f^\alpha}, \quad \text{where } \alpha \in [1.8, 2.2]$$

In log-log coordinates, this relationship is linear:

$$\log P(f) = \log A - \alpha \log f$$

### 2.2 Azimuthal Radial Integration
To compute the 1D radial energy profile $P(r)$, GUARD integrates the centered 2D power spectrum over concentric discrete rings of radius $r$:

$$r(u, v) = \sqrt{(u - u_0)^2 + (v - v_0)^2}$$

$$P(r) = \frac{1}{|\Omega_r|} \sum_{(u, v) \in \Omega_r} |F_{\text{shifted}}(u, v)|^2$$

$$\Omega_r = \left\{(u, v) \;\middle|\; r - \frac{1}{2} \le r(u, v) < r + \frac{1}{2}\right\}$$

### 2.3 Goodness-of-Fit Coefficient ($R^2$)
Fitting the linear model $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ across mid-to-high radial frequencies $r \in [r_{\min}, r_{\max}]$ (where $x_i = \log r_i$ and $y_i = \log P(r_i)$), the coefficient of determination is:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^M (y_i - \hat{y}_i)^2}{\sum_{i=1}^M (y_i - \bar{y})^2}$$

- **Authentic Camera Media**: $R^2 \ge 0.94$ (smooth power-law roll-off).
- **Synthetic Diffusion / GANs**: $R^2 < 0.92$ (flat high-frequency tail or anomalous spectral spikes).

### 2.4 High-Frequency Spike Ratio
To detect discrete grid artifacts from strided transposed convolutions (upconvolution checkerboard effect), GUARD measures localized spectral peaks in the upper 30% of the frequency band:

$$\text{SpikeRatio} = \frac{\sum_{r \ge 0.70 r_{\max}} \mathbb{I}\left(P(r) > 1.35 \cdot \hat{P}_{\text{trend}}(r)\right)}{\sum_{r \ge 0.70 r_{\max}} 1}$$

where $\mathbb{I}(\cdot)$ is the indicator function.

---

## 3. Stochastic Camera Sensor Noise Residual Modeling

### 3.1 Poisson-Gaussian Sensor Noise Model
Physical digital image sensors (CMOS/CCD) convert incoming photons into electrons, subject to quantum shot noise and electronic read noise:

$$I(x, y) = I_0(x, y) + \eta_p(I_0(x, y)) + \eta_g(x, y)$$

- **Photon Shot Noise**: $\eta_p \sim \mathcal{P}(\lambda \cdot I_0)$, signal-dependent Poisson process.
- **Read Noise**: $\eta_g \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$, thermal and circuit noise.

The overall noise variance is affine in intensity:

$$\text{Var}(I \mid I_0) = a \cdot I_0 + b$$

### 3.2 Noise Residual Extraction via Spatial Decomposition
Generative models generate images via continuous latent vectors without modeling optical sensor grain, resulting in unnaturally smooth local pixel neighborhoods.

GUARD extracts the high-frequency residual $R(x, y)$ by subtracting a locally adaptive non-linear smoothing filter $\hat{I}$:

$$\hat{I}(x, y) = \text{Median}_{3 \times 3}(I(x, y))$$

$$R(x, y) = I(x, y) - \hat{I}(x, y)$$

### 3.3 Sample Variance Formulation
$$\bar{R} = \frac{1}{HW} \sum_{x=1}^H \sum_{y=1}^W R(x, y)$$

$$\sigma_n^2 = \frac{1}{HW - 1} \sum_{x=1}^H \sum_{y=1}^W (R(x, y) - \bar{R})^2$$

- **Natural Camera Sensor Grain**: $\sigma_n^2 \ge 1.5$.
- **Synthetic Latent Smoothing**: $\sigma_n^2 < 1.0$.

---

## 4. Deep Neural Spatial Stream & Squeeze-and-Excitation Attention

### 4.1 Mobile Inverted Bottleneck Convolution (MBConv)
The Spatial Stream uses an **EfficientNet-B4** backbone. A standard convolution on an input tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{\text{in}}}$ with kernel size $K$ has computational complexity:

$$\text{FLOPs}_{\text{standard}} = H \cdot W \cdot K^2 \cdot C_{\text{in}} \cdot C_{\text{out}}$$

MBConv decomposes this into a Depthwise Separable Convolution:
1. **Pointwise Expansion ($1 \times 1$)**: Expands channels by expansion factor $t$: $C_{\text{in}} \to t C_{\text{in}}$.
2. **Depthwise Convolution ($K \times K$)**: Convolves each channel independently.
3. **Pointwise Projection ($1 \times 1$)**: Projects back to $C_{\text{out}}$.

$$\text{FLOPs}_{\text{MBConv}} = H \cdot W \cdot t C_{\text{in}} \cdot \left(K^2 + C_{\text{out}}\right)$$

$$\text{Complexity Reduction Ratio} \approx \frac{1}{C_{\text{out}}} + \frac{1}{K^2}$$

### 4.2 Squeeze-and-Excitation (SE) Channel Attention
SE blocks recalibrate channel feature maps by explicitly modeling inter-channel dependencies.

1. **Squeeze (Global Information Aggregation)**:
   $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i, j), \quad \mathbf{z} \in \mathbb{R}^C$$
2. **Excitation (Adaptive Channel Dependency Modeling)**:
   $$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \mathbf{z})\right)$$
   where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, reduction ratio $r=4$, and $\sigma(\cdot)$ is the sigmoid function.
3. **Channel-Wise Scale Modulation**:
   $$\tilde{\mathbf{x}}_c = s_c \cdot \mathbf{x}_c$$

---

## 5. Swin Transformer Shifted Window Attention

### 5.1 Standard Multi-Head Self-Attention Complexity
Given feature map $h \times w$ patches with embedding dimension $C$:

$$\Omega(\text{Standard Multi-Head Self-Attention}) = 4hwC^2 + 2(hw)^2C$$

Because complexity scales quadratically with $(hw)^2$, standard self-attention is computationally prohibitive on high-resolution forensic images ($380 \times 380$ or $512 \times 512$).

### 5.2 Windowed Multi-Head Self-Attention (W-MSA)
Swin Transformer partitions the image into non-overlapping windows of size $M \times M$ ($M=7$):

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

For $hw = 56 \times 56$ and $M=7$, W-MSA achieves a $\approx 64\times$ reduction in computational operations.

Within each window, attention with relative position bias $\mathbf{B}$ is computed as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right)\mathbf{V}$$

where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{M^2 \times d}$ and relative position bias $\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$ models spatial coordinates within local windows.

---

## 6. Dynamic Gated Attention Cross-Modal Fusion

### 6.1 Gating Mechanism Formulation
Given spatial features $\mathbf{f}_s \in \mathbb{R}^{1792}$ and frequency features $\mathbf{f}_f \in \mathbb{R}^{512}$:

1. **Projection into Common Embedding Space**:
   $$\mathbf{h}_s = \text{LayerNorm}(\text{GELU}(\mathbf{W}_s \mathbf{f}_s + \mathbf{b}_s)) \in \mathbb{R}^{1024}$$
   $$\mathbf{h}_f = \text{LayerNorm}(\text{GELU}(\mathbf{W}_f \mathbf{f}_f + \mathbf{b}_f)) \in \mathbb{R}^{1024}$$
2. **Joint Representation**:
   $$\mathbf{j} = [\mathbf{h}_s \,\|\, \mathbf{h}_f] \in \mathbb{R}^{2048}$$
3. **Dynamic Attention Gate**:
   $$\mathbf{\alpha} = \sigma\left(\mathbf{W}_{g2} \cdot \text{GELU}(\mathbf{W}_{g1} \mathbf{j} + \mathbf{b}_{g1}) + \mathbf{b}_{g2}\right) \in [0, 1]^{1024}$$
4. **Modulated Fusion with Residual Bypass**:
   $$\mathbf{j}_{\text{gated}} = [\mathbf{\alpha} \odot \mathbf{h}_s \,\|\, (1 - \mathbf{\alpha}) \odot \mathbf{h}_f] + \mathbf{j} \in \mathbb{R}^{2048}$$
   $$\mathbf{f}_{\text{fused}} = \text{GELU}(\mathbf{W}_o \mathbf{j}_{\text{gated}} + \mathbf{b}_o) \in \mathbb{R}^{1024}$$

### 6.2 Mathematical Proof of Guaranteed Gradient Flow
Let $\mathcal{L}$ be the objective loss function. The gradient backpropagating to the spatial representation $\mathbf{h}_s$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_s} = \frac{\partial \mathcal{L}}{\partial \mathbf{j}_{\text{gated}}} \cdot \frac{\partial \mathbf{j}_{\text{gated}}}{\partial \mathbf{h}_s}$$

Examining the Jacobian:

$$\frac{\partial \mathbf{j}_{\text{gated}}}{\partial \mathbf{h}_s} = \text{diag}(\mathbf{\alpha}) + \mathbf{I}_{1024} + \mathbf{h}_s \cdot \left(\frac{\partial \mathbf{\alpha}}{\partial \mathbf{h}_s}\right)^T$$

Because of the identity residual bypass $\mathbf{I}_{1024}$ and $\mathbf{\alpha} \in [0, 1]$:

$$\left\|\frac{\partial \mathbf{j}_{\text{gated}}}{\partial \mathbf{h}_s}\right\| \ge 1.0, \quad \left\|\frac{\partial \mathbf{j}_{\text{gated}}}{\partial \mathbf{h}_f}\right\| \ge 1.0$$

**Conclusion**: Gradient starvation is mathematically impossible; both feature streams continue to learn throughout training regardless of gate values.

---

## 7. Gradient-Weighted Class Activation Mapping (Grad-CAM)

To provide interpretable spatial evidence overlays, GUARD computes class-specific activation maps for synthetic classification.

### 7.1 Channel Importance Weight Formulation
For class $c \in \{\text{Synthetic}, \text{Authentic}\}$ with logit output $y^c$, the importance weight $\alpha_k^c$ for convolutional feature map channel $\mathbf{A}^k \in \mathbb{R}^{U \times V}$ is:

$$\alpha_k^c = \frac{1}{U \cdot V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$

This represents the global average pooling of gradients with respect to feature activations.

### 7.2 Rectified Activation Map & Alpha Blending
The localization map is computed via a linear combination of activation maps followed by a Rectified Linear Unit ($\text{ReLU}$) to isolate features that positively contribute to the synthetic decision:

$$L_{\text{Grad-CAM}}^c(i, j) = \text{ReLU}\left(\sum_k \alpha_k^c A_{i, j}^k\right)$$

The map is bilinearly upsampled to input resolution $(H, W)$ and normalized:

$$L_{\text{norm}}^c = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + \epsilon}$$

Alpha blending overlays the colormap onto the original RGB image:

$$I_{\text{overlay}} = (1 - \beta) \cdot I_{\text{RGB}} + \beta \cdot \text{Colormap}\left(L_{\text{norm}}^c\right)$$

where $\beta \in [0.10, 0.90]$ is the user-tunable blending alpha.

---

## 8. Temporal Video Forensics & Scene-Cut Filtering

### 8.1 Normalized L1 Pixel Difference
Given consecutive video frames $I_t, I_{t-1} \in \mathbb{R}^{H \times W \times 3}$:

$$\Delta L_1(t) = \frac{1}{3 \cdot H \cdot W} \sum_{c=1}^3 \sum_{x=1}^H \sum_{y=1}^W \left|I_t(c, x, y) - I_{t-1}(c, x, y)\right|$$

### 8.2 2D HSV Histogram Correlation
Color distribution shifts indicate scene changes. Frames are converted to HSV space, and normalized 2D Hue-Saturation histograms $\mathbf{H}_t \in \mathbb{R}^{50 \times 60}$ are extracted:

$$\rho(t) = \frac{\sum_i (H_t(i) - \bar{H}_t)(H_{t-1}(i) - \bar{H}_{t-1})}{\sqrt{\sum_i (H_t(i) - \bar{H}_t)^2 \sum_i (H_{t-1}(i) - \bar{H}_{t-1})^2}}$$

### 8.3 Discrete 2D Laplacian Texture Variance
The discrete Laplacian operator evaluates structural edge variance across high-contrast contours:

$$\nabla^2 I(x, y) = I(x+1, y) + I(x-1, y) + I(x, y+1) + I(x, y-1) - 4 I(x, y)$$

$$\text{Lap}(t) = \text{Var}\left(\nabla^2 I_t\right) = \frac{1}{HW} \sum_{x, y} \left(\nabla^2 I_t(x, y) - \mu_{\nabla^2}\right)^2$$

### 8.4 Scene-Cut Aware Temporal Flicker Metric
Editorial jump cuts are distinguished from generative morphing:

$$\text{Cut}(t) = \left(\Delta L_1(t) \ge 0.38\right) \lor \left(\rho(t) \le 0.45\right)$$

$$\Delta s_t = |s_t - s_{t-1}|, \quad \Delta \text{Lap}_t = \frac{|\text{Lap}(t) - \text{Lap}(t-1)|}{\max_k \text{Lap}(k) + \epsilon}$$

$$\text{Flicker}(t) = \begin{cases} 0.0, & \text{if } \text{Cut}(t) = \text{True} \\ \Delta s_t + 0.40 \cdot \Delta \text{Lap}_t, & \text{otherwise} \end{cases}$$

### 8.5 Top-K Anomaly Aggregation
To prevent brief deepfake appearances (e.g., a 2-second face-swap in a 60-second video) from being diluted by averaging, GUARD computes the mean probability across the top 20% most anomalous frames:

$$\mathcal{S}_{\text{Top-K}} = \frac{1}{K} \sum_{i=1}^K s_{(i)}, \quad K = \max\left(1, \lfloor 0.20 \cdot N \rfloor\right)$$

where $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(N)}$ is the sorted sequence of frame AI probabilities.

---

## 9. Optimization, Loss Functions & Regularization

### 9.1 Numerically Stable Binary Cross-Entropy Loss
Given raw logit $z \in \mathbb{R}$ and ground truth label $y \in \{0, 1\}$:

$$P(\text{Fake}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\mathcal{L}_{\text{BCE}}(z, y) = -y \log \sigma(z) - (1 - y) \log (1 - \sigma(z))$$

To prevent arithmetic overflow for large $|z|$:

$$\mathcal{L}_{\text{BCE}}(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$

### 9.2 Decoupled Weight Decay (AdamW)
Unlike standard Adam, where $L_2$ weight decay is added directly to gradients (distorting momentum):

$$\mathbf{g}_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})$$

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \frac{\eta_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and weight decay $\lambda = 10^{-4}$.

### 9.3 Cosine Annealing Learning Rate Schedule with Warmup
For epoch $t \in [1, T_{\max}]$:

$$\eta_t = \begin{cases} \eta_{\min} + \frac{t}{T_{\text{warm}}} (\eta_{\max} - \eta_{\min}), & t \le T_{\text{warm}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left[1 + \cos\left(\frac{t - T_{\text{warm}}}{T_{\max} - T_{\text{warm}}} \pi\right)\right], & t > T_{\text{warm}} \end{cases}$$

---

## 10. Decision Theory, Threshold Calibration & Forensic Metrics

### 10.1 Tunable Decision Boundary
Given predicted probability $p \in [0.0, 1.0]$ and calibrated threshold $\tau \in [0.10, 0.90]$:

$$\hat{y} = \begin{cases} 1 \text{ (Synthetic / Deepfake)}, & p \ge \tau \\ 0 \text{ (Authentic Media)}, & p < \tau \end{cases}$$

- **Strict Realism ($\tau = 0.65$)**: Minimizes False Positives ($\text{FPR} \to 0$); protects authentic photography.
- **Balanced Profile ($\tau = 0.50$)**: Standard Bayes-optimal decision threshold under equal class priors.
- **Aggressive Detection ($\tau = 0.35$)**: Prioritizes Recall ($\text{FNR} \to 0$); catches subtle AI artwork, posters, and Midjourney renders.

### 10.2 Confusion Matrix Formulations
Given True Positives ($\text{TP}$), True Negatives ($\text{TN}$), False Positives ($\text{FP}$), and False Negatives ($\text{FN}$):

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall (TPR)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{False Positive Rate (FPR)} = \frac{\text{FP}}{\text{TN} + \text{FP}}$$

$$\text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 10.3 Area Under the ROC Curve (AUC-ROC)
Computed by integrating True Positive Rate as a function of False Positive Rate across all thresholds $\tau \in [0, 1]$:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(u)) \, du$$

Equivalently, using the Wilcoxon-Mann-Whitney U statistic:

$$\text{AUC} = \frac{1}{N_{\text{pos}} \cdot N_{\text{neg}}} \sum_{i=1}^{N_{\text{pos}}} \sum_{j=1}^{N_{\text{neg}}} \left[\mathbb{I}(p_i^+ > p_j^-) + \frac{1}{2}\mathbb{I}(p_i^+ = p_j^-)\right]$$

### 10.4 Equal Error Rate (EER)
The operational threshold $\tau^*$ where False Positive Rate equals False Negative Rate:

$$\tau^* = \arg\min_\tau |\text{FPR}(\tau) - \text{FNR}(\tau)|$$

$$\text{EER} = \frac{\text{FPR}(\tau^*) + \text{FNR}(\tau^*)}{2}$$

---

## Summary Reference Table

| Mathematical Concept | Core Equation / Formulation | Forensic Purpose |
| :--- | :--- | :--- |
| **2D Fast Fourier Transform** | $F(u, v) = \frac{1}{\sqrt{HW}} \sum f(x, y) e^{-i 2\pi (ux/H + vy/W)}$ | Isolates high-frequency synthesis patterns from spatial content. |
| **Radial Power-Law Decay** | $P(r) \propto r^{-\alpha}, \quad R^2 \ge 0.94$ | Distinguishes natural optical roll-off from synthetic frequency flattening. |
| **Sensor Shot Noise Residual** | $\sigma_n^2 = \text{Var}(I - \text{Median}_{3\times 3}(I))$ | Quantifies physical CMOS/CCD optical grain vs. synthetic smoothing. |
| **Gated Dynamic Attention** | $\mathbf{\alpha} = \sigma(\mathbf{W}_2 \text{GELU}(\mathbf{W}_1 \mathbf{j})) \implies \|\nabla_{\mathbf{h}_s} \mathbf{j}_{\text{gated}}\| \ge 1.0$ | Dynamic weighting across compression levels with zero gradient starvation. |
| **Scene-Cut Filter** | $\text{Cut}(t) = (\Delta L_1 \ge 0.38) \lor (\rho_{\text{HSV}} \le 0.45)$ | Excludes human video editing transitions from temporal anomaly scoring. |
| **Laplacian Variance** | $\text{Lap} = \text{Var}(\nabla^2 I)$ | Tracks temporal edge sharpness stability across video frames. |
| **Top-K Anomaly** | $\mathcal{S}_{\text{Top-K}} = \frac{1}{K} \sum_{i=1}^K s_{(i)}$ | Detects localized face-swaps in extended video streams. |

