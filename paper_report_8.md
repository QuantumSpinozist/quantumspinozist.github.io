
# Paper Report 8: Segment Anything

**Paper**: [https://arxiv.org/pdf/2304.02643#cite.he2022masked](https://arxiv.org/pdf/2304.02643#cite.he2022masked)

## Introduction

This project by META AI presents a new foundation model for image segmentation called Segment Anything Model (SAM).
To achieve this they define a general segmentation task in which the image segmentation model is prompted with points, bounding boxes, segmentation masks
or natural text in an interactive way. Like with LLMs this enables practicioners to adapt SAM to their specific problem via prompt engineering.

The authors use their model in an data engine annotation loop to assemble an extremely large segmentation dataset called SA-1B containing 11M images with 1B masks (the largest to date).


## Method

### Task

Like in NLP the model is tasked with producing a segmentation mask that is valid for the given prompt.
To reiterate, in this context a prompt will be one or any combination of the following:
a set of points, a bounding box, a mask or text. This also suggests a pre-training scheme. Given
an image with ground truth masks, one generates a number of prompts and validates wether the produced model output
is a valid result for the given prompt.

### Model architecture

The model consists of the following components: a powerful image encoder backbone, a prompt encoder and a segmentation mask decoder.
The image encoder is a MAE pretrained ViT encoder, specifically a ViT-H/16 from the original paper.

We distinguish between the mask prompts, which we call dense, and all other prompts which are called sparse. The sparse prompts are
all mapped to 256 dimensional latent vectors. A point is encoded as the sum of a location encoding and one of two learned representations for either foreground
or background. The box is the location encoding of the upper left corner plus a learned representation for "upper left corner" combined with the same for
the lower rigt corner. For text the CLIP encoder is used. For mask prompts a kind of CNN produces a 256 dim vector that is simply added to the image encoding.

The decoder takes in the image encoding (which already contains the mask prompts) and a set of prompt encodings and outputs three segmentation masks. It
is purposefully desgined to be rather lightweight. We will not cover the specific details and refer interested readers to the papers appendix. A linear combination
of focal and dice loss is used as the objective.

### Data Engine

The dataset construction process using the data engine is made up of three stages. The first stage, called the assised-manual stage, is essentially standard interactive
segmentation annotation with weak early version of SAM. Annotaters prompt the model on new images and correct the produced masks manually. In this stage 4.3M masks over 120k images
are produced.

The second stage is called semi-automatic. To diversify the segmentation masks the model first automatically detects easy masks. To do this the authors trained a generic object detector
(so only one class that just represents all objects) based
on the first stage data, that is then used to prompt the model. These images with confident masks are then handed to annotators who identify less obvious objects. In this stage the average
number of masks per image increases from 44 to 72.

At this point the model is already very capable. The final stage is fully automatic, the new images are prompted by a regular grid of points. From the resulting masks, the confident and stable ones are chosen
and filtered for duplicates. In the end over 1B masks over 11M images are produced this way (so over 99% of all masks come from this stage).

### Dataset

To ensure data quality, 500 images with around 50k masks are sampled and given to annotators. The images are quite high resolution when compared to other datasets (lower side set to 1500).
Looking at the average mask centers suggests that the mask positions are more diverse than in previous datasets, which show a much stronger center bias.

The authors also perform detailed investigations to confirm the dataset is in accordance with ethical guidelines. They show that different geographic locations and income groups are represented fairly.
They also use the More Inclusive Annotations for People (MIAP) dataset to show that SAM is also fair with respect to segmentation of different faces.

## Zero Shot Transfer Experiments

The authors perform a number of experiments to test the models performance. All except the first one constitute a significant form of transfer learning and are all zero shot, namely:
edge detection, object proposal, instance segmentation, (pure) text-to-mask. The authors start with an evaluation of the segmentation abilities with a single point prompt.
They compare SAM to RITM on 23 datasets. SAM outperforms RITM consistently both in IoU and segmentation quality (as rated by human annotators for a subset of the data) on all datasets.

For edge detection the model is prompted by a regular grid of points. The resulting mask probability maps (withou applying threshold) are than passed through a Sobel filter to obtain edges.
SAM produces reasonable edge detections and performs pretty much on par with specialized methods like HED on BSDS-500.

SAM, evaluated for zero-shot object proposal generation on LVIS, outperforms ViTDet-H on medium, large, rare, and common objects despite lacking LVIS-specific training.
However, SAM falls somewhat short on small and frequent objects, highlighting the advantage of ViTDet-H's dataset-specific tuning.

Instance segmentation can be done analogously to the fully automated stage in the data engine.
In zero-shot instance segmentation on COCO and LVIS, SAM underperforms ViTDet in mask AP scores but generates visually superior masks with crisper boundaries, as validated by human ratings. This suggests SAM’s masks may be of higher perceptual quality, while ViTDet leverages dataset-specific biases, particularly on COCO, due to its training.

To perform effective text to mask inference SAM is specifically trained with the CLIP image encodings of all sufficiently large masks as text prompts.
This works because CLIPs contrastive loss ensures that text and image latent vectors are close to each other. So SAM can use both image and text encodings from either CLIP encoder
as prompts. Qualitative results show that SAM performs this task reasonably well.

## Conclusion

SAM presents a very powerful foundation model for segmentation and vision as a whole. Through the prompting capability the model is highly flexible and can be used effectively without finetuning.
It will be interesting to see if these kinds of big foundation models will ultimately become fully ubiquitous. From what I know, specialized models are still SOTA in domains like biomedical imaging, but this might change
in the coming years.
