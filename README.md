# [Reproducability Challenge 2021] Free Lunch for Few-Shot Learning: Distribution Calibration

FREE LUNCH FOR FEW-SHOT LEARNING: Distribution Calibration written
by Shuo Yang, Lu Liu, Min Xu is to transfer statistics from base classes with
enough examples to calibrate the distribution of these few-sample classes, and
then to draw a sufficient number of examples from the calibrated distribution to
expand the input of the classifier. The calibrated distribution is then drawn from
a sufficient number of examples to expand the input to the classifier Yang et al.
(2021). By running the Distribution Calibration code in the appendix of this paper

and pre-training the data, we will confirm whether the results mitigate the overfit-
ting phenomenon in few-sample learning, as claimed in this paper. By calculating

the accuracy of SVM and logistic regression, Tukey transformation and the pres-
ence or absence of generated features, we see that Distribution Calibration does

have some improvement on the overfitting problem.



## Original paper 

- link: https://openreview.net/forum?id=JWOiYxMG92s

## Requirements

- numpy==1.17.2
- matplotlib==3.1.1
- tqdm==4.36.1
- torchvision==0.6.0
- torch==1.5.0
- Pillow==7.1.2


### You can directly download the extracted features/pretrained models from the link:
https://drive.google.com/drive/folders/1IjqOYLRH0OwkMZo8Tp4EG02ltDppi61n?usp=sharing

After downloading the extracted features, please adjust your file path according to the code.


## Evaluate our distribution calibration

To evaluate our distribution calibration method, run:

```eval
python evaluate_DC.py
```
## Team members

- Yijun    Chen             yc16g20@soton.ac.uk
- Shuning  Ling             sl4m20@soton.ac.uk
- Pasinpat Vitoochuleechoti pv1u20@soton.ac.uk

## Original Code

https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration



