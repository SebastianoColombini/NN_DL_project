# How we got here

A short account of the path that led to the final models, written by the two authors. The
[report](report.pdf) tells the same story in full, with the numbers (Section III).

## 1. Reading

We started from a close reading of the base paper, *Deep Visual Attention Prediction*
([Wang & Shen, 2018](https://arxiv.org/abs/1705.02544)), and a few additional papers:

- Ghariba et al., [*Visual Saliency Prediction Based on Deep Learning*](https://www.mdpi.com/2078-2489/10/8/257)
  (Information, 2019): a VGG-16 encoder-decoder with 5 conv and 5 deconv blocks.
- Yan et al., [*Review of Visual Saliency Prediction*](https://www.mdpi.com/2076-3417/12/1/309)
  (Applied Sciences, 2022): used for the report and the evaluation measures.
- Lou et al., [*TranSalNet*](https://www.sciencedirect.com/science/article/pii/S0925231222004714)
  (Neurocomputing, 2022): saliency prediction with transformers.
- Kümmerer et al., [*DeepGaze II*](https://arxiv.org/abs/1610.01563) (2016): object recognition
  and visual saliency are closely related, so a good saliency network resembles an
  object-recognition network.
- [IEEE 10331919](https://ieeexplore.ieee.org/abstract/document/10331919): training CNNs well at
  low cost.
- [IEEE 11003418](https://ieeexplore.ieee.org/abstract/document/11003418): how sound influences
  visual saliency (interesting rather than directly useful).

Two conclusions drove the design: saliency networks look a lot like object-recognition
networks, and transformers are a promising next step for the encoder.

## 2. Beta: the paper, as is

`notebooks/archive/beta.ipynb` follows the architecture of the base paper faithfully and
trains with an MSE loss, as suggested in the course presentation. Unlike the paper we start
from an untrained network, and the results were poor: geometric artefacts and the smoothing
typical of MSE training.

## 3. v1.0: modern training, a better loss

`notebooks/archive/v1.0.ipynb` replaces the dated techniques (Nesterov momentum) with Adam,
adds batch normalisation (absent from the original model) and, most importantly, introduces a
deep-supervision loss that combines **KL divergence, correlation coefficient and
similarity** on the final map and on each decoder branch. This produced a large jump, both in
the loss and visually. A small informal study group of 7 people agreed that the maps looked
right.

## 4. v1.1: rectangular input, early stopping

`notebooks/v1.1.ipynb` keeps the architecture (hence 1.1, not 2.0), makes the input
rectangular (240×320), enlarges it and adds early stopping. Only marginal improvements.

## 5. v2.0: Swin-T encoder

`notebooks/v2.0.ipynb` swaps the CNN encoder for a Swin-Tiny transformer pre-trained on
ImageNet. Expectations were high; even after a very long training run (probably limited by
a low learning rate) the results are about the same as v1.1, with marginal gains.

## 6. Comparison

Both models were tested on MIT1003, which is never used for training, to measure how well
they generalise. Average losses are very close and the maps are visually almost
indistinguishable.
