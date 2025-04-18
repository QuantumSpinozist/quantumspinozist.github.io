# Paper Report 11: End-to-end Recovery of Human Shape and Pose

**Paper** [https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf)

## Introduction

In human pose estimation the goal is usually to infer the 2D or 3D positions of a predefined set of joints from an image. In this paper this situation is extended
to Human Mesh Recovery (HMR) where a full 3D mesh of the human body is reconstructed. Two objectives are defined to achieve that: first reconstruct a mesh that conforms to
the shown image, second make sure the mesh defines a plausible human body. For the former a reprojection loss of selected keypoints (usually the joints) corresponding to the mesh is minimized, for the later
one uses adversarial training where a second model tries to judge whether a mesh is valid based on large dataset of human meshes.

## Preliminary: The SMPL
The 3D mesh of the human body is encoded using the Skinned Multi-Person Linear Model (SMPL) (the original paper "SMPL: A Skinned Multi-Person Linear Model" is at [https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)). 
In the standard version we have $N=6890$ vertices and $K=23$ joints. An SMPL will consists of the following:

- A set of $N$ vectors $\mathbf{\bar T} \in \mathbb{R}^{3N}$ defining the positions of the vertices in the zero pose
- A blend weight matrix $\mathcal{W} \in \mathbb{R}^{N \times K}$ defining the influence of eah joint on each mesh vertex
- A blend shape function $B_S(\boldsymbol{\beta}) : \mathbb{R}^{|\boldsymbol{\beta}|} \mapsto \mathbb{R}^{3N}$ that maps the vector of shape parameters $\boldsymbol{\beta}$ defining the individual body shape to offsets in the vertex positions
- Similarly the joint regressor function $J(\boldsymbol{\beta}) : \mathbb{R}^{|\boldsymbol{\beta}|} \mapsto \mathbb{R}^{3K}$ that maps the shape parameters $\boldsymbol{\beta}$ to the joint positions
- A pose blend function $B_P(\boldsymbol{\theta}) : \mathbb{R}^{|\boldsymbol{\theta}|} \mapsto \mathbb{R}^{3N}$ mapping pose parameters $\boldsymbol{\theta}$ (the zero pose is $\boldsymbol{\theta}^*$) to vertex offsets
- Finally a skinning function $\mathcal{W}(\mathbf{T}, \mathbf{J}, \boldsymbol{\theta}, \mathcal{W}) : \mathbb{R}^{3N \times 3K \times |\boldsymbol{\theta}| \times |\mathcal{W}|} \mapsto \mathbb{R}^{3N}
$ (either linear or dual quaternion) that deforms mesh according to the rotated joints

The overall model is then given by $M(\boldsymbol{\beta}, \boldsymbol{\theta}) = \mathcal{W}(T_P(\boldsymbol{\beta}, \boldsymbol{\theta}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathcal{W})
$ where $T_P(\boldsymbol{\beta}, \boldsymbol{\theta}) = \mathbf{\bar T} + B_S(\boldsymbol{\beta}) + B_P(\boldsymbol{\theta})$. A standard skinning function is used which we will not conver in detail (refer to the original paper or the literature).

The shape blend function is linear and defined by $B_S(\boldsymbol{\beta}) = \sum_{n=1}^{|\boldsymbol{\beta}|}\beta_n \mathbf{S_n}$ where $\mathbf{S_n}\in \mathbb{R}^{3N}$ are the orthonormal PCA components of the shape displacements.

For the pose blend functions we first define $R: \mathbb{R}^{|\boldsymbol{\theta}|} \mapsto \mathbb{R}^{9K}$ which maps the pose to a vector of flattened relative rotation matrices (for this reason $3\cdot3 = 9$).
With this we get $B_P(\boldsymbol{\theta}) = \sum_{n=1}^{9K} (R_n(\boldsymbol\theta) - R_n(\boldsymbol\theta^*)) \boldsymbol P_n$ where $\boldsymbol P_n$ are the blend shape vectors, similar to the shape blend function.

The joint regressor is linear using a learned matrix $\mathcal{J}$ that is applied to $\mathbf{\bar T} + B_S(\boldsymbol{\beta})$. Overall we end up with the parameters $\Phi = (\mathbf{\bar T}, \mathcal{W}, \mathcal{P}, \mathcal{S}, \mathcal{J})$ that need to be learned.

## Model

First we sketch the overall model pipeline. The image is fed through a CNN based encoder (ResNet-50) resulting in a representation $\phi$ from which we want to infer $\Theta = (\boldsymbol \theta, \boldsymbol \beta, R, t, s)$.
Here the first two paramter vectors correspond to the SMPL shape and pose which we discussed in the last section, while the latter three define the camera (view angle, position, scale). Based on $\Theta$ we project the keypoints
onto the image and compare them with the 2D ground truth labels. The reconstruction of $\Theta$ is done in an iterative way as we will see in a moment.
As the second component of the pipeline an adversarial discriminator is trained to distinguish between the reconstructed SMPL parameters and the "real" ones we get from a database. This regularizes the SMPL, biasing the model
towards parameters that conform to plausible meshes. Overall we get the loss

$$ L = \lambda (L_{reprojection} + L_{3D;optional}) + L_{adversarial} $$

where the second term is used when 3D information is available during reconstruction, and the $\lambda$ denotes a loss weight hyperparameter.

### Iterative 3D Regression with Feedback

Doing the reconstruction directly is not possible with reasonable accuracy, instead the authors use an iterative approach. The regression module (a two layer MLP) takes in $\phi$ and the current estimate $\Theta_t$ and produces
a correction for $\Theta$. In their implementation this is repeated three times. The reprojection loss uses the L1 metric. For the optional 3D loss, that is used when Motion capture data is available, the L2 losses of comparing the
3D positions of the joints as well as the parameter vectors are added.


### Factorized Adversarial Prior

Since SMPL is so explicit we can decompose or factorize the discriminator, making training much easier. Specifically the authors use a discriminator for the shape and again decompose the hiearchical kinematic tree structure of the
pose parameters to get individual discriminators for each joint. Overall they end up with K+2 small networks. Because of this and since we have the additional reprojection loss (avoiding mode collapse) the adversarial training
does not suffer from the usual problems one gets with GANs.

