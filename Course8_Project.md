# Practical Machine Learning Course Project

### Synopsis
The project uses data gathered from 6 participants who were asked to perfom one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions.  Accelerometers were placed on the belt, forearm, arm, and dumbell of each participant.  The manner in which each exercise was performed was categorized into the following classes as detailed below:  
* Class A -- performed according to the specification  
* Class B -- throwing the elbows to the front  
* Class C -- lifting the dumbbell only halfway  
* Class D -- lowering the dumbbell only halfway  
* Class E -- throwing the hips to the front  
  
A model was then developed to predict how well a participant executes a particular excercise using the classes listed above.  
Reference: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

### Exploratory Data Analysis


```r
## First load the necessary libraries
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
## Download the test and train data sets if not already done
if(!file.exists("Course8/pml-training.csv")){
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    filename <- "Course8/pml-training.csv"
    download.file(url, filename)
    }

if(!file.exists("Course8/pml-testing.csv")){
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    filename <- "Course8/pml-testing.csv"
    download.file(url, filename)
    }
```

Load the train and test sets. Note that some missing values in the csv file are simply empty cell values, whereas others are "NA", so they are both specified in the na.strings paramter in the read.csv function below.


```r
train_raw <- read.csv("Course8/pml-training.csv", header=TRUE, stringsAsFactors = TRUE, na.strings = c(NA,""))
test_raw <- read.csv("Course8/pml-testing.csv", header=TRUE, stringsAsFactors = TRUE, na.strings = c(NA,""))
```
After doing some initial exploratory analysis of the train data set, it can be seen that there are several columns where over 95% of the values are missing.  Remove those columns from the training data since they will not be valuable for building a predictive model.


```r
training <- train_raw[ ,(colSums(is.na(train_raw))/nrow(train_raw)<0.95)]
```

Now get rid of the columns used to identify observations (e.g. timestamps) that aren't useful for building the model


```r
training[c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')] <- list(NULL)
```

### Model Building
Fit a Random Forests model to the training data.  
Use trainControl() to set the cross validation method to "cv".

```r
modfit <- train(classe ~ ., method="rf", data=training, trControl=trainControl(method="cv"), number=3)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

The model is described below:

```r
modfit
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17661, 17659, 17660, 17659, 17661, 17658, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9956169  0.9944555
##   27    0.9954133  0.9941981
##   52    0.9905725  0.9880744
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Predict "classe" for the 20 observations in the test data set

```r
pred <- predict(modfit, test_raw)
```

The most important variables are listed below:

```r
varImp(modfit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## yaw_belt               84.87
## magnet_dumbbell_z      73.43
## pitch_belt             62.67
## magnet_dumbbell_y      62.41
## pitch_forearm          61.36
## roll_forearm           51.96
## magnet_dumbbell_x      51.74
## accel_dumbbell_y       46.10
## accel_belt_z           44.62
## magnet_belt_z          43.14
## magnet_belt_y          42.99
## roll_dumbbell          42.52
## accel_dumbbell_z       36.47
## roll_arm               34.60
## accel_forearm_x        33.20
## total_accel_dumbbell   30.13
## gyros_belt_z           29.49
## yaw_dumbbell           29.36
## accel_arm_x            28.56
```

Given the high importance of the top 7 variables, re-fit the model using only those variables

```r
modfit2 <- train(classe ~ roll_belt + pitch_forearm + yaw_belt + pitch_belt + magnet_dumbbell_y + magnet_dumbbell_z + roll_forearm, method="rf", data=training, trControl=trainControl(method="cv"), number=3)
```

Predict "classe" for the 20 observations in the test data set using this simpler model

```r
pred2 <- predict(modfit2, test_raw)
pred_compare <- (pred == pred2)
pred_compare
```

```
##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [15] TRUE TRUE TRUE TRUE TRUE TRUE
```
The results in the vector above show that both models predict the same classe for each test observation

This simpler model is described below:

```r
modfit2
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17659, 17659, 17661, 17660, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.9879727  0.9847916
##   4     0.9887371  0.9857569
##   7     0.9842014  0.9800200
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 4.
```

```r
modfit2$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, number = 3) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 4
## 
##         OOB estimate of  error rate: 0.99%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5542   18   16    3    1 0.006810036
## B   18 3733   33   11    2 0.016855412
## C    8   17 3384   13    0 0.011104617
## D    0    3   16 3191    6 0.007773632
## E    0   15    8    7 3577 0.008317161
```
### Conclusion
This simpler model (modfit2) is still very accurate with an out of sample error rate of less than 1%.  Fitting the model with fewer features requires much less processing time, and the number of input variables randomly chosen at each split (mtry) is reasonable as well.  The simpler model accurately predicted all 20 test cases, and should be the final model selected.
