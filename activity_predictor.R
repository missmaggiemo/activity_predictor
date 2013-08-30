#This code predicts the activity that a participant was performing while wearing
#a Samsung Galaxy S2 smartphone. The activities are WALKING, WALKING_UPSTAIRS,
#WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

#The data comes from here: http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones .

#Updated 2013-08-30

#Load data. If you have not downloaded the data, please see the README file.

load("./samsungData.rda")

names(samsungData)

length(names(samsungData))

#All columns are data from the Samsung phone except [562] "subject" and [563]
#"activity"

#NAs?

sum(is.na(samsungData))

#No NAs! Woo hoo!

sum(duplicated(names(samsungData)))

#There are some repeats in the names of the columns.

for(name in unique(names(samsungData))){
  if(length(which(names(samsungData)==name)) > 1){
    print(name)
    print(which(names(samsungData)==name))
  }
}

#Are the columns themselves repeats?

sum(duplicated(samsungData))

#No, they are not simply repeats. We need to rename them. Moreover, the names
#are messy and full of "()" and ",", so let's rename them all.

names <- names(samsungData)

names <- gsub("\\()", "", names); names <- gsub("\\(", "_", names); names <- gsub("\\)", "", names)

#Repeats are at "bandsEnergy" labels, names[303:344], names[382:423], names[461:502].

for(i in 1:3){
  for(j in 1:length(names)){
    if(names[j] %in% names[j+1:length(names)]){
      names[j] <- paste(names[j], as.character(i), sep="_")
    }
  }
}

gsub("_1_2", "_2", names) -> names

names(samsungData) <- names

names(samsungData)

#Now repeats are different, and names are tidy-er.

#Also, let's make sure that this data is in a nice data frame.

samsungData <- data.frame(samsungData)

#Separate the data into "Train", "Valudation", and "Test" sets.

class(samsungData$subject)

unique(samsungData$subject)

median(samsungData$subject)

length(unique(samsungData$subject))

#The prompt says: 
#Your task is to build a function that predicts what activity a subject is
#performing based on the quantitative measurements from the Samsung phone. For
#this analysis your training set must include the data from subjects 1, 3, 5,
#and 6.  But you may use more subjects data to train if you wish. Your test set
#is the data from subjects 27, 28, 29, and 30, but you may use more data to
#test. Be careful that your training/test sets do not overlap.

#My Train set is:

samsungTrain <- samsungData[samsungData$subject < 12,]

unique(samsungTrain$subject)

dim(samsungTrain)

#My Validation set is:

samsungValid <- samsungData[samsungData$subject > 12,]
samsungValid <- samsungValid[samsungValid$subject < 23,]

unique(samsungValid$subject)

dim(samsungValid)

#After fitting the data to the Train set, I'll apply my model to the Validation set
#and tweak it if I need to.

#My Test set is:

samsungTest <- samsungData[samsungData$subject > 22,]

unique(samsungTest$subject)

dim(samsungTest)

#After I apply my model to the Validation set and finish any tweaking, I'll apply my
#model ONCE to my Test set.

#I'm choosing to have a Train, Validation, and Test set to ensure that I don't
#overfit my model.


#And now I'll only work with the training data, samsungTrain.

set.seed(306131)

library(tree)

tree <- tree(as.factor(activity) ~., data=samsungTrain)

summary(tree)

plot(tree); text(tree)

plot(cv.tree(tree, FUN=prune.tree, method="misclass"))

#With this method, the best model is a tree with 9 nodes, so "tree" is optimal. 

#Use randomForest method (see http://stat-www.berkeley.edu/users/breiman/RandomForests/).

set.seed(306132)

library(randomForest)

forest <- randomForest(samsungTrain[1:561], as.factor(samsungTrain$activity))

print(forest)

#Make "forest" better?

tuneRF(samsungTrain[1:561], as.factor(samsungTrain$activity)) #Tune the forest, leaving out "subject" as a variable.

#"forest" is already using mtry=23, so I won't bother building another model
#based on that.

forest.cv <- rfcv(samsungTrain[1:561], as.factor(samsungTrain$activity))

with(forest.cv, plot(n.var, error.cv, log="x", pch=19, xlab="Number of Variables Used", ylab="CV Error", main="Cross-Validation Error vs. Number of Variables"))


plot = ggplot(forestcv.frame, aes(x = n.var, y = error.cv)) + 
    geom_point(aes(size=40)) + 
    scale_x_log10(breaks = c(1,2,5,10,50,100,300,500)) +
    theme_bw() +
    theme(legend.position = "none", 
          axis.text.x = element_text(size = 40),
          axis.title.x = element_text(size = 40), 
          axis.text.y = element_text(size = 40),
          axis.title.y = element_text(size = 40),
          plot.title = element_text(size = 40)) +
    labs(list(title="Cross-Validation Error vs. Number of Variables", x="Number of Variables Used", y="CV Error"))

jpeg(filename = "Forest CV.jpg", height = 1000, width = 1000, pointsize = 20, quality = 100)
plot
dev.off()

#The more variables, the better?

set.seed(306134)

forest.tuned = randomForest(samsungTrain[1:561], as.factor(samsungTrain$activity), mtry = 92)

print(forest.tuned)


#Just to be sure that we're not overfitting, let's get the most important variables:

varImpPlot(forest)

importance(forest) -> import

import.names = data.frame(name = names(samsungTrain[,1:561]), import = as.numeric(import[1:561,]))

import.names = import.names[order(import.names$import, decreasing=TRUE),]

most_import = import.names[1:50,1] #50 most important variables

#Forest with 10 most important variables, in an effort not to overfit the data:

set.seed(306133)

forest.rerun <- randomForest(samsungTrain[,most_import], as.factor(samsungTrain$activity))

print(forest.rerun)

tuneRF(samsungTrain[,most_import], as.factor(samsungTrain$activity))

#"forest.rerun" is already using mtry=6, so I won't bother building another
#model based on that.

#Let's compare the three models-- "tree", "forest", and "forest.rerun".

summary(tree)

print(forest)

print(forest.rerun)

#Pay attention to error rates. "forest" has the lowest error.

#Which does better with the validation data?

#My function for assessing the misclassification rate of each model's predition.

missClassToo = function(values,prediction){
  wrong=0L;
  for(i in 1:length(values)){
    if(values[i] != prediction[i]){
      wrong=wrong+1
    }
  }
  misclass = wrong/length(values)
  return(misclass)
}

#My function for creating a confidence interval for the misclassification error:

BinomError95 <- function(p, n){
  interval = sqrt(p*(1-p)/n)*1.96 
  Confint = c((p+interval), (p-interval)) 
  return(Confint)
}


#"predBlank", a prediction that all activities are "laying", will be the benchmark.

predBlank <- rep("laying", length(samsungValid$activity))

table(predBlank, samsungValid$activity)

missClassToo(samsungValid$activity, predBlank) -> blankError

blankError

BinomError95(blankError, length(samsungValid$activity))

#"tree": one standard tree.

predict(tree, samsungValid, type="class") -> predTree

table(predTree, samsungValid$activity)

missClassToo(samsungValid$activity, predTree) -> treeError

treeError

BinomError95(treeError, length(samsungValid$activity))

#"forest": a randomForest.

predict(forest, samsungValid, type="class") -> predForest

table(predForest, samsungValid$activity)

missClassToo(samsungValid$activity, predForest) -> forestError

forestError

BinomError95(forestError, length(samsungValid$activity))


#forest.tuned: a rerun of randomForest sampling 92 variables at each split.

predict(forest.tuned, samsungValid, type="class") -> predForest.tuned

table(predForest.tuned, samsungValid$activity)

missClassToo(samsungValid$activity, predForest.tuned) -> tunedError

tunedError

BinomError95(tunedError, length(samsungValid$activity))

#"forest.rerun": a rerun of randomForest using only the most important variables.

predict(forest.rerun, samsungValid, type="class") -> predForest.rerun

table(predForest.rerun, samsungValid$activity)

missClassToo(samsungValid$activity, predForest.rerun) -> rerunError

rerunError

BinomError95(rerunError, length(samsungValid$activity))


#All prediction models are better than the benchmark.

predFunction <- c("predBlank", "predForest", "predForest.rerun", "predForest.tuned"); 
misclassRate <- c(missClassToo(samsungValid$activity, predBlank), 
  missClassToo(samsungValid$activity, predForest), 
  missClassToo(samsungValid$activity, predForest.tuned),
  missClassToo(samsungValid$activity, predForest.rerun));
misclassRate = data.frame(predFunction, misclassRate)

plot = ggplot(misclassRate, aes(x=predFunction, y= misclassRate)) +
geom_bar(aes(fill=predFunction)) +
theme_bw() +
theme(axis.text.x = element_text(size = 40), 
    axis.text.y = element_text(size = 40),
    legend.position="none",
    panel.grid.minor=element_blank(), 
    panel.grid.major=element_blank(),
    rect = element_blank()) +
xlab("") + ylab("") 

jpeg(filename = "Misclassification Rate.jpg", height = 1000, width = 1500, pointsize = 20, quality = 100)
plot
dev.off()


#"forest" is the best model.


#Run the model on the test data set: THE MOMENT OF TRUTH.

predict(forest, samsungTest, type="class") -> predTEST

table(predTEST, samsungTest$activity)

missClassToo(samsungTest$activity, predTEST) -> TESTerror

TESTerror

BinomError95(TESTerror, length(samsungTest$activity))


#For reference, all "laying":

predBlank2 <- rep("laying", length(samsungTest$activity))

table(predBlank2, samsungTest$activity)

missClassToo(samsungTest$activity, predBlank2) -> blank2Error

blank2Error

BinomError95(blank2Error, length(samsungTest$activity))



#Final figure:

par(mfrow=c(1,2))

with(forest.cv, plot(n.var, error.cv, log="x", pch=19, xlab="Number of Variables Used", ylab="CV Error", main="Cross-Validation Error vs. Number of Variables"))

barplot(misclassRate, names.arg=c("all laying", "tree", "forest", "forest.rerun"), main="Misclassification Rate for Each Model")

#Caption: On the left, cross-validation error for random forest model “forest”
#is graphed against the number of variables utilized in the model. The
#cross-validation error decreases as more variables are added to the model. This
#shows why “forest”, with 561 variables included, is a more accurate model than
#“forest.rerun”, with only 10 variables included. On the right, the
#misclassification rate for each model used to predict from the validation data
#the activity performed from the accelerometer data: “all laying”, “tree”,
#“forest”, and “forest.rerun”. “forest” has the lowest misclassification rate.


#To illustrate the accuracy of "forest":

par(mfrow=c(1,2))

plot(samsungTest$angle_X.gravityMean, samsungTest$angle_Y.gravityMean, col=as.factor(predTEST), main="Predicted Activity", xlab="angle_X.gravityMean", ylab="angle_Y.gravityMean")
with(samsungTest, legend(x=c(-1, 0),y=c(-0.5,-1), ncol=2, x.intersp=0.3, y.intersp=1, legend=unique(activity), col=unique(as.factor(activity)), pch=1, cex=0.8))

with(samsungTest, plot(angle_X.gravityMean, angle_Y.gravityMean, col=as.factor(activity), main="Real Activity", xlab="angle_X.gravityMean", ylab="angle_Y.gravityMean"))
with(samsungTest, legend(x=c(-1, 0),y=c(-0.5,-1), ncol=2, x.intersp=0.3, y.intersp=1, legend=unique(activity), col=unique(as.factor(activity)), pch=1, cex=0.8))

#There's only one problem: This model is amazingly over-fit.


#An attempt at correcting for over-fitting:


#Create a correlation matrix.

M = cor(data.frame(samsungTrain[,1:562], as.numeric(as.factor(as.vector(samsungTrain$activity)))))

#Choose only the variables with at least a |correlation value with activity| = 0.8.

which(M[,563] >= 0.8 | M[563,] <= -0.8) -> list

#That list included the activity variable itself, so we take that out.

list = list[1:37]

list

set.seed(306135)

#Create a random forest model using only those variables.

forest_lessfit = randomForest(samsungTrain[,list], as.factor(samsungTrain$activity))

predTEST = predict(forest_lessfit, samsungTest, type="class")

table(predTEST, samsungTest$activity)

TESTerror = missClassToo(samsungTest$activity, predTEST)

TESTerror

BinomError95(TESTerror, length(samsungTest$activity))

#This model doesn't do as well with the test data, but the real test would be
#applying it to more data. I would expect this model to win out when applied
#to many more data sets.

par(mfrow=c(1,2))

plot(samsungTest$angle_X.gravityMean, samsungTest$angle_Y.gravityMean, col=as.factor(predTEST), main="Predicted Activity", xlab="angle_X.gravityMean", ylab="angle_Y.gravityMean")
with(samsungTest, legend(x=c(-1, 0),y=c(-0.5,-1), ncol=2, x.intersp=0.3, y.intersp=1, legend=unique(activity), col=unique(as.factor(activity)), pch=1, cex=0.8))

with(samsungTest, plot(angle_X.gravityMean, angle_Y.gravityMean, col=as.factor(activity), main="Real Activity", xlab="angle_X.gravityMean", ylab="angle_Y.gravityMean"))
with(samsungTest, legend(x=c(-1, 0),y=c(-0.5,-1), ncol=2, x.intersp=0.3, y.intersp=1, legend=unique(activity), col=unique(as.factor(activity)), pch=1, cex=0.8))
