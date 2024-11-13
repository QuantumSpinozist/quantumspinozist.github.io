# Paper Report 5: DeepPose: Human Pose Estimation via Deep Neural Networks

**Paper**: [https://arxiv.org/pdf/1312.4659](https://arxiv.org/pdf/1312.4659)

## Introduction
I am working on a PyTorch reimplementation of this paper at [DeepPose](https://github.com/QuantumSpinozist/Deep-Pose) (work in progress).

DeepPose presents one of the earliest successful attempts of using deep learning for human pose estimation (HPE). It also provides an interesting blueprint for how the deep learning revolution transformed many application problems, especially in domains like computer vision. Before deep learning, HPE required a combination of complex handcrafted features, statistical models, and optimization techniques. In contrast, DeepPose's deep learning approach is much simpler, while significantly outperforming previous results.

## Methodology

For a pose model with $$ k $$ body joints, we define the pose vector 
$$  \mathbf{y} = (\mathbf{y}_1^\top, \ldots,\mathbf{y}_i^\top ,\ldots ,\mathbf{y}_k^\top )^\top \in \mathbb{R}^{2k} $$ with $$ \mathbf{y}_i = (x_i, y_i)^\top $$
denoting the position of the $$i$$-th joint. We also introduce notation for normalizing a pose vector using a bounding box $$ b = (b_c, b_w, b_h) $$ where $$b_c$$ is
the box center, $$ b_w $$ the width, and $$ b_h $$ the height, namely

$$ 
N(\mathbf{y}_i; b) = 
\begin{pmatrix}
\frac{1}{b_w} & 0 \\
0 & \frac{1}{b_h}
\end{pmatrix}
(\mathbf{y}_i - b_c).
$$

For an input image $$x$$, we use $$ N(\mathbf{x}; b) $$ to denote the cropped image according to the bounding box. If no $$b$$ is given, we normalize to the whole image
(i.e., the bounding box is the image itself). Our neural net is supposed to learn a function $$ \psi(\mathbf{x};\theta) = y$$ that assigns a (normalized) pose vector to each image.
Since the function returns a normalized pose vector, we train it on a normalized dataset $$ D_N = \{ (N(\mathbf{x}), N(\mathbf{y})) | (\mathbf{x}, \mathbf{y}) \in D \} $$ ($$D$$ is the original dataset)
using an $$ L_2 $$ loss as our learning objective

$$ \arg\min_{\theta} \sum_{(x,y) \in D_N} \sum_{i=1}^{k} \left\| \mathbf{y}_i - \psi_i(x; \theta) \right\|_2^2 .$$

The neural network is a simple 7-layer CNN. If you wish to know the precise architecture, you may refer to the illustration in the original paper.

This setup works as a rudimentary model for HPE, but the resulting performance is limited. To improve this, the pose vector is used as a starting point for a cascade of
the same architecture nets that iteratively estimate a displacement from the previous pose vector. Specifically, the recursion is:

$$\text{Stage} \, s: \quad \mathbf{y}_i^{s} \leftarrow \mathbf{y}_i^{(s-1)} + N^{-1} \left( \psi_i(N(x; b); \theta_s); b \right)$$

$$\text{for} \quad b = b_i^{(s-1)}$$

$$b_i^s \leftarrow (\mathbf{y}_i^s, \sigma \text{diam}(\mathbf{y}^s), \sigma \text{diam}(\mathbf{y}^s))$$

for every stage $$ s=1, \ldots, S $$ and every joint $$i = 1,\ldots,k $$. Here, $$\text{diam}(\mathbf{y})$$ is a specific measure for the distance between opposing joints and is dataset dependent, while
the scaling factor $$\sigma$$ is a hyperparameter that needs to be chosen.
The network no longer outputs the full pose vector but just the 2-dimensional position of the joint. We also normalize each joint vector independently and use additional data augmentation

$$ D_A^s = \left\{ (N(x; b), N(\mathbf{y}_i; b)) \mid 
(x, \mathbf{y}_i) \in D, \, \delta \sim \mathcal{N}_i^{(s-1)}, \, 
b_{i,c} \leftarrow b_{i,c} +\delta 
\right\}. $$

The Gaussian for the augmentation is calibrated on the training data accordingly. The new training objective for stage $$s$$ can then formally be expressed as:

$$ \theta_s = \arg\min_{\theta} \sum_{(x,\mathbf{y}_i) \in D_N} \left\| \mathbf{y}_i - \psi_i(x; \theta) \right\|_2^2 .$$

## Evaluation

The model pipeline was trained and tested on the Frames Labeled In Cinema (FLIC) and Leeds Sports Dataset (LSP). Two different metrics were used to evaluate the results, namely the Percentage of Correct Parts (PCP) and the Percentage of Detected Joints (PDJ). 

While I will not delve deeply into the specifics of the evaluations, it is clear that DeepPose outperformed previous baselines. The stage-one model without cascading performed on par with existing baselines but slightly underperformed against the best baselines. However, after incorporating the second and third stages, the performance showed clear improvements, with DeepPose reaching a PCP score of $$0.69$$ compared to the next-best baseline at $$0.64$$.

## Conclusion

DeepPose was a pivotal work in applying deep learning to human pose estimation, setting a new standard for the field.
The paper showcased how neural networks could perform as good as traditional feature-based methods.
The introducing of a cascade of networks for refined pose estimation further enabled a significant outperformance of those previous approaches.
This paved the way for subsequent advances in the field. Although its architecture is simple by modern standards, DeepPose remains a foundational contribution to human pose estimation.




