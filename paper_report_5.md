# Paper Report 5: DeepPose: Human Pose Estimation via Deep Neural Networks

**Paper**: [https://arxiv.org/pdf/1312.4659](https://arxiv.org/pdf/1312.4659)

## Introduction

DeepPose presents one of the earliest successful attempts of using deep learning for human pose estimation (HPE).
I find that it also presents an interesting blue print for how the deep learning revolution changed so many application problems
(especially in domains like computer vision). Before deep learning HPE required a combination of complex handcrafted features, statistical models
and optimization. If we compare this with DeepPose, the deep learning approach is much simpler while also outperforming previous reults significantly.

## Methodology

For a pose model with $$ k $$ body joints we define the pose vector 
$$  \mathbf{y} = (\mathbf{y}_1^\top, \ldots,\mathbf{y}_i^\top ,\ldots ,\mathbf{y}_k^\top )^\top \in \mathbb{R}^{2k} $$ with $$ \mathbf{y}_i = (x_i, y_i)^\top $$
denoting the position of the $$i$$-th joint. We also introduce notation for normalizing a pose vector using a bounding box $$ b = (b_c, b_w, b_h) $$ where $$b_c$$ is
the box center, $$ b_w $$ the width and $$ b_h $$ the height, namely

$$ 
N(\mathbf{y}_i; b) = 
\begin{pmatrix}
\frac{1}{b_w} & 0 \\
0 & \frac{1}{b_h}
\end{pmatrix}
(\mathbf{y}_i - b_c).
$$

For an input image $$x$$ we use $$ N(\mathbf{x}; b) $$ to denote the cropped image according to the bounding box. If no $$b$$ is given we normalize to the whole image
(i.e. the bounding box is the image itself). Our neural net is supposed to learn a function $$ \psi(\mathbf{x};\theta) = y$$ that assigns a (normalized) pose vector to each image.
Since the function returns a normalized pose vector we train it on a normalized dataset $$ D_N = \{ (N(\mathbf{x}), N(\mathbf{y})) | (\mathbf{x}, \mathbf{y}) \in D \} $$ ($$D$$ is the original dataset)
using an $$ L_2 $$ loss as our learning objective

$$ \arg\min_{\theta} \sum_{(x,y) \in D_N} \sum_{i=1}^{k} \left\| \mathbf{y}_i - \psi_i(x; \theta) \right\|_2^2 .$$

The neural network is a pretty simple 7 layer CNN. If you wish to know the precise architecture you may look at the illustration on the preview page (or even better in the original paper of course).

This set up already works as a rudimentary model for HPE, but the resulting performance will be pretty lacking. To improve this the pose vector is used as a starting point for an cascade of
same architecture nets that iteratively estimate a displacement from the previous pose vector. Concretely we use the recursion

$$\text{Stage} \, s: \quad \mathbf{y}_i^{s} \leftarrow \mathbf{y}_i^{(s-1)} + N^{-1} \left( \psi_i(N(x; b); \theta_s); b \right)$$

$$\text{for} \quad b = b_i^{(s-1)}$$

$$b_i^s \leftarrow (\mathbf{y}_i^s, \sigma \text{diam}(\mathbf{y}^s), \sigma \text{diam}(\mathbf{y}^s))$$
