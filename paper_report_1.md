# Paper Report 1: Does Knowledge Distillation Really Work?

**Paper**: [https://arxiv.org/pdf/2106.05945](https://arxiv.org/pdf/2106.05945)

## Introduction

Knowledge distillation is a widely adopted technique for transferring knowledge from a large teacher model to a smaller student model in a given task. This paper closely examines how this process actually happens, focusing on student-teacher agreement. The authors observe a large agreement gap for sufficiently difficult problems and identify the key reason for this issue.

## Initial Observation

The paper starts with the observation that transferring a predictive distribution does not work exceptionally well in knowledge distillation for settings that are more difficult than toy datasets like MNIST. They achieve close to 100% test agreement between the teacher and student model trained on MNIST (using LeNet-5 for self-distillation). However, both self-distillation and distillation of a small ensemble on CIFAR-100 (using ResNet-56) fundamentally change the situation. Now, the test agreement does not surpass 80%, even with extensive use of synthetic training data. This might seem puzzling at first, since the model trivially has the capacity to replicate the distribution (recall that one experiment uses self-distillation). The article systematically uncovers the reasons for this gap.

## Hypotheses and Experiments

The paper proposes six possible explanations for the agreement gap, which are ruled out through experiments:

1. **Student Capacity**: Since they use self-distillation, this is already implausible, but they further show that increasing the model's capacity does not improve agreement.
2. **Architecture**: Changing the model from ResNet to VGG produces similar results.
3. **Dataset Scale and Complexity**: The same phenomenon is observed for ImageNet.
4. **Domain**: They also experiment with sentiment analysis (IMDB), showing similar results.
5. **Identifiability**: The problem could lie in the type of data the model is distilled on. Matching the teacher's predictions on the chosen distillation dataset might not be sufficient to match her performance in general.
6. **Optimization**: The optimization problem may not be solved well.

Explanations 5 and 6 are covered in their own sections of the article. For the problem of identifiability, they experiment with extensive use of data augmentation and attempt to distill the model on a sample from the training data distribution that the teacher model has not seen. None of these approaches significantly increase agreement. In contrast, changing the temperature of the logits has a much greater effect and is unrelated to identifiability, which seems to rule out reason 5 as a key factor in their view.

Explanation 6, however, seems promising. They show that the agreement on the distillation dataset is significantly lower than 100% when the data contains samples that are new for the teacher. In the earlier experiments, it was shown that test agreement increases when more out-of-distribution (OOD) data is added for distillation, but this simultaneously lowers training agreement. Therefore, there is a trade-off between making the student better at matching the teacher on OOD examples and making the training optimization problem too difficult by adding too much OOD distillation data. 

They also provide an illustrative experiment where they initialize the student near the teacher's weights during self-distillation. It becomes clear that the student converges to a suboptimal attractor when the initial weights are too far from the teacher's.

## Discussion and Comments

The main takeaway is that optimization during knowledge distillation is usually very difficult and highly dependent on the distillation data. The actual knowledge transfer does not work as well as we might think, even for expressive student models. The fact that the student can still achieve good generalization shows that common intuitions about how distillation works may be inaccurate.

A potential criticism of the paper (as raised by many reviewers) is that the conclusions do not quite fit the title. Some of the statements may also be too general, as the experiments mostly focus on CIFAR-100 and the ResNet architecture. Additionally, I found some of the theoretical sections to lack clarity (especially the beginning of section 5.2). Lastly, many of the provided plots were difficult to interpret; tabular presentations would have been preferable in some cases.

After reading the paper, the central question I had was: how can we improve optimization in knowledge distillation? The authors provide good arguments for why the agreement gap should not be ignored, even if distillation can often work well in practice.


