
# Paper Report 8: Segment Anything

**Paper**: [https://arxiv.org/pdf/2304.02643#cite.he2022masked](https://arxiv.org/pdf/2304.02643#cite.he2022masked)

## Introduction

This project by META AI presents a new foundation model for image segmentation called Segment Anything Model (SAM).
To achieve this they define a general segmentation task in which the image segmentation model is prompted with points, bounding boxes, segmentation masks
or natural text in an interactive way. Like with LLMs this enables practicioners to adapt SAM to their specific problem via prompt engineering.

The authors use their model in an data engine to assemble an extremely large segmentation dataset called SA-1B containing 11M images with 1B masks (the largest to date).


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


### Dataset

## Zero Shot Transfer Experiments
