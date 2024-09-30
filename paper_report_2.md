# Paper Report 2: Hybrid Active Learning via Deep Clustering for Video Action Detection

**Paper**: [https://tinyurl.com/hybridclaus](https://tinyurl.com/hybridclaus)

## Introduction


## Hybrid Active Learning and CLAUS

Previous active learning (AL) approaches for video action detection have either used inter-sample selection or intra-sample selection.
For inter-sample selection videos are chosen and then fully labelled while for intra-sample selection there is at least one
annotated frame for each video, and new frames are chosen in each AL step. The new hybrid AL approach combines the two. In each iteration
new videos and frames are chosen.

For each video informative frames are identified by calculating the pixel level uncertainty of the model over $R$ runs using Monte Carlo Dropout
(so some parts of the net are randomly switched off during inference) with 

$$U = \frac{1}{R}\sum_{k=0}^R -\log(M(p, k))$$

where $M(p,k)$ is the model prediction value at pixel $p$ and MC dropout run $k$. Now the top $A$ frames are identified and the video gets an informativeness score of

$$ V_{score} = \frac{1}{A} \sum_{f=1}^A \sum_{p} U_{f, p}.$$

There is an additional mechanism to disinsentivise the AL loop from picking frames that are close to each other since this creates redundancy.

To further improve the diversity of the selection the authors use CLAUS. The model learns a k-means cluster representation of the videos using deep clustering.
At each iteration we pick videos from each cluster proportional to its size.

## Spatio-Temporal Weighted Loss
To effectively use the limited number of labelled frames available for each video the authors propose a new kind of loss, called STeW.
First pseudolabels are created by interpolating between the labelled frames. Due to temporal consistency the pseudolabels close to labelled frames
will be more trustworthy on average. To encode this effectively in the learning scheme each pseudolabelled frame gets a pixelwise weight that is calculated as the product of
the distance to the next labelled frame and the mean of the pixel value over the next and last $W$ neighboring frames

$$ \phi_f (p) = \min_{f_a} Dist(f, f_a) \cdot \frac{1}{W+1} \sum_{w=f-W}^{w=f+W} x(f, p).$$

The mean pixel value is put to 1 when it is not betwen two given values. This makes it so that consistent values get a high score, while pixels that change in the given
temporal region tend to receive lower weights. The overall loss is then simply the weighted mean over all frames of the regular BCE.


## Results

The authors show that their method outperforms comparable active learning techniques and even achieves comparable performance to a model
trained using 90% annotation with just 5%. Using ablation studies they also demonstrate that both CLAUS and STeW improve the performance.

