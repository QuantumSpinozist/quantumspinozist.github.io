# Paper Report 2: Hybrid Active Learning via Deep Clustering for Video Action Detection

**Paper**: [https://tinyurl.com/hybridclaus](https://tinyurl.com/hybridclaus)

## Introduction

This paper presents a novel active learning method designed to drastically reduce the number of annotated frames required for video action detection.

## Hybrid Active Learning and CLAUS

Previous active learning (AL) approaches for video action detection have employed either inter-sample selection or intra-sample selection. In inter-sample selection, entire videos are chosen and fully labeled, while in intra-sample selection, at least one frame is annotated for each video, and new frames are selected during each AL step. The proposed hybrid AL approach combines both strategies: in each iteration, new videos and frames are selected.

For each video, informative frames are identified by calculating the pixel-level uncertainty of the model over $$R$$ runs using Monte Carlo Dropout (where parts of the network are randomly deactivated during inference). The uncertainty is defined as:


$$U = \frac{1}{R}\sum_{k=0}^R -\log(M(p, k))$$

where $$M(p,k)$$ represents the model's prediction at pixel $$p$$ during the $$k$$-th Monte Carlo Dropout run. The top $$A$$ frames are then identified, and the video is assigned an informativeness score:

$$ V_{score} = \frac{1}{A} \sum_{f=1}^A \sum_{p} U_{f, p}.$$

Additionally, a mechanism is implemented to discourage the AL loop from selecting frames that are temporally close to each other, which would introduce redundancy.

To further enhance diversity in frame selection, the authors employ CLAUS. The model learns a k-means cluster representation of the videos through deep
clustering. During each iteration, videos are selected from each cluster proportionally to its size.

## Spatio-Temporal Weighted Loss
To make efficient use of the limited number of labeled frames in each video, the authors propose a novel loss function called Spatio-Temporal Weighted (STeW) loss. First, pseudolabels are generated by interpolating between the labeled frames. Due to temporal consistency, pseudolabels closer to the labeled frames tend to be more reliable. To account for this in the learning process, each pseudolabeled frame receives a pixel-wise weight, calculated as the product of the distance to the nearest labeled frame and the mean pixel value over the next and previous $$W$$ neighboring frames:

$$ \phi_f (p) = \min_{f_a} Dist(f, f_a) \cdot \frac{1}{W+1} \sum_{w=f-W}^{w=f+W} x(f, p).$$

The mean pixel value is set to 1 if it falls outside of a specified range. This ensures that consistent pixel values are assigned higher scores,
while pixels that vary over the temporal region receive lower weights.
The overall loss is then computed as the weighted mean of the standard Binary Cross-Entropy (BCE) loss across all frames.


## Results

The authors demonstrate that their method outperforms comparable active learning techniques and achieves performance comparable to a model trained with 90% annotation, using only 5% annotation. Ablation studies further show that both CLAUS and STeW contribute to the improved performance.

