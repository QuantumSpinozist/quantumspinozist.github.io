# Paper Report 11: End-to-end Recovery of Human Shape and Pose

**Paper** [https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf)

## Introduction

In human pose estimation the goal is usually to infer the 2D or 3D positions of a predefined set of joints from an image. In this paper this situation is extended
to Human Mesh Recovery (HMR) where a full 3D mesh of the human body is reconstructed. Two objectives are defined to achieve that: first reconstruct a mesh that conforms to
the shown image, second make sure the mesh defines a plausible human body. For the former a reprojection loss of the joints corresponding to the mesh is minimized, for the later
one uses adversarial training where a second model tries to judge whether a mesh is valid based on large dataset of human meshes.

## Preliminary: The SMPL

## Model


### Iterative 3D Regression with Feedback

### Factorized Adversarial Prior
