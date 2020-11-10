# LSTM neural network pipeline for evolution forecast of COVID-19.

![Image](https://www.compromisorse.com/upload/noticias/027/27772/shutterstock_1654034092_lh.jpg)

## **Introduction.**

### Problem description:

The main idea of the project is to train several recurrent neural networks to be able to predict the evolution of Covid19 as accurately as possible. This will help us to make better decisions about population control. Training and testing dataset is from USA.

### Original Model.

From this differntial equation:

- <img src="https://latex.codecogs.com/gif.latex? $$\partial_tf = f(t)(N - f(t))\text{ growth equation }$$ " /> 

we can deduce the solution that we use to do the sigmoidal regression. This regression is the most used for this type of analisis.

- <img src="https://latex.codecogs.com/gif.latex? $$f(t) = \frac {N}{1 + e^{-k(t - t_0)}}\text {solution}$$ " />


This model is very useful but there is some limitations, especialy in the fluctuations of the growth.

##**Metodology.**

### LSTM neural Network Model

This option is better to predict the evolution of the disease because the prediction is adjusted according to the previous data. Additionally this model is easy to train.

![Image](https://www.researchgate.net/profile/Savvas_Varsamopoulos/publication/329362532/figure/fig5/AS:699592479870977@1543807253596/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell.jpg)

### Pipeline dependencies:

- numpy
- pandas
- pytorch
- matplotlib
- datetime
- scipy
- sklearn
- streamlit

### Pipeline componets:

- 3 modules (preprocesing, analisis, models)
- 1 LSTM class
- 2 train neural networks
- 3 analisis functions
- 4 preprocesing functions


