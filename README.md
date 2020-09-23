# Primal-Dual Fair Meta-learning (PDFM)

A PyTorch implementation of "A Primal-Dual Subgradient Approach for Fair Meta Learning" (ICDM 2020).

<div style="text-align:center"><img src ="overview-PDFM.png" ,width=200/></div>

### Abstract
<p align="justify">
The problem of learning to generalize on unseen classes during the training step, also known as few-shot classification, has attracted considerable attention. Initialization based methods, such as the gradient-based model agnostic meta-learning (MAML), tackle the few-shot learning problem by ``learning to fine-tune”. The goal of these approaches is to learn proper model initialization, so that the classifiers for new classes can be learned from a few labeled examples with a small number of gradient update steps. Few shot meta-learning is well-known with its fast-adapted capability and accuracy generalization onto unseen tasks. Learning fairly with unbiased outcomes is another significant hallmark of human intelligence, which is rarely touched in few-shot meta-learning.  In this work, we propose a novel Primal-Dual Fair Meta-learning framework, namely PDFM, which learns to train fair machine learning models using only a few examples based on data from related tasks. The key idea is to learn a good initialization of a fair model's primal and dual parameters so that it can adapt to a new fair learning task via a few gradient update steps. Instead of manually tunning the dual parameters as hyperparameters via a grid search, PDFM optimizes the initialization of the primal and dual parameters jointly for fair meta-learning via a subgradient primal-dual approach. We further instantiate an example of bias controlling using decision boundary covariance (DBC) as the fairness constraint for each task, and demonstrate the versatility of our proposed approach by applying it to classification on a variety of three real-world datasets. Our experiments show substantial improvements over the best prior work for this setting. </p>

## BibTeX

```
@inproceedings{zhao2020pdfm,
  title={A Primal-Dual Subgradient Approach for Fair Meta Learning},
  author={Zhao, Chen and Chen, Feng and Wang, Zhuoyi and Khan, Latifur},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  year={2020},
  organization={IEEE}
}
```
