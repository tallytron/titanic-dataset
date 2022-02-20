# titanic-dataset

## PREDICTING PASSENGER FATE ON THE TITANIC

## Introduction

On April 10th, 1912, a passenger liner named RMS Titanic, or more commonly known solely as Titanic, departed from Southampton, England travelling to New York City, USA with roughly 2,224 passenger and crew members on board. On April 14th the ship hit an iceberg and sank on April 15th killing at least 1,500 people. There are several accounts of what transpired on the ship after it hit the iceberg, however one thing that was for sure is that there were only enough lifeboats for half the capacity of the ship resulting in certain passengers surviving over others. Our goal is to understand whether there were certain socio-economic factors that contributed to whether a person survived or died.

## Data

To build a model to predict whether a passenger or crew member on the Titanic survived, I participated in the Kaggle competition, “Titanic - Machine Learning from Disaster”. In this competition I was provided with two datasets, training, and testing. There were 12 data elements given with the mix of categorical and continuous data types in the training dataset. The testing set provided all variables excluding the predictor variable. As a result, I used the testing dataset as our validation set to base our predictions on.

## EDA 

As the first step to understand the dataset, I explored the variables provided. The dataset contained various socio-economic features such as the fare the passenger paid, the age of the passenger, what class they were seated on the Titanic, if there was a family accompanying the passenger, and the passenger’s gender. Before I built the model, I explored the socio-economic features visually for the preliminary understanding of this data. As a first step, I aggregated the data to perform some initial data exploration and plotted them as shown.

![image](https://user-images.githubusercontent.com/80222038/154824775-42aae79a-1859-48b9-b20b-d8f81ed4f481.png)

The graph on the top left, depicts the Survival Rate by Gender. From this graph, we can see that more females survived as compared to males. When we see the passenger’s survival rate by the class they travelled on the titanic, the graph on the top right shows the proportion of the passengers that survived by their class. Here we could see that first class or second-class passengers survived at a higher rate over third-class passengers. The passenger’s class can to some extent attribute to the amount of wealth they hold.

The bottom graph on the left shows the survival and death breakdown by family size. Here we can see that smaller families survived at a higher rate than larger families. However, we can also see that solo passengers did not have a better survival rate. This could partially be due to families with younger children being given priority over individuals on their own. It should be noted that this chart was created after we completed data pre-processing.

Finally, on the bottom right graph, we were able to further understand the survival rate by gender and class to show if multiple factors may have impacted whether a person survived or died. We could see that most of the females in the first and second classes survived however, for males, the survival rate decreased per change in class.

These plots help support our objective and gives cause to perform further data processing to predict whether socio-economic characteristics had an impact on whether a passenger.

### Pre-processing done for the data

![image](https://user-images.githubusercontent.com/80222038/154824827-81c7d562-002c-4bd3-877f-79e5273c38d6.png)

Once the pre-processing was done, model building begins. For this competition, I was looking to predict a binary outcome of 0 or 1 for death and survival. I explored the three different models to make predictions. Each model was developed using the full dataset after pre-processing. I decided to use this approach to perform the preliminary evaluation and to help build a baseline to help explain our objective. 

## Generalized Linear Model

The first model, a simple generalized linear regression model, was built with the pre-processed Train data to assess the significance of predictors. This model had an AUC score of 0.884 and AIC score of 783.25 against the actual survival result. Next, this initial model was then used to predict the survival result in the revised Test data, and then results were submitted to Kaggle to assess the performance resulting in a score of 0.77033. Even though his model provided the highest AUC score, the prediction did not perform well. This could be due to overfitting of the data during training. Since the test set provided by Kaggle did not include the predictor variable, we could not use this to test our model. Because of this, the AUC score recorded was based on our training set.



