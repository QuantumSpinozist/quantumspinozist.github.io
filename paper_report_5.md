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
N(\mathbf{y}_i; \mathbf{b}) = 
\begin{pmatrix}
\frac{1}{b_w} & 0 \\
0 & \frac{1}{b_h}
\end{pmatrix}
(\mathbf{y}_i - \mathbf{b}_c)
$$



The HPE pipeline starts with object detection, so every input image $$ \mathbf{x} $$ is assigned a bounding box $$ b = (b_c, b_w, b_h) $$ where $$b_c$$ is
the box center, $$ b_w $$ the width and $$ b_h $$ the height. 
