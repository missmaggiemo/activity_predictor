# README

I created this code for Prof. Jeff Leek's Data Analysis Coursera course.

[Here is my Coursera profile.](https://www.coursera.org/user/i/62da43c330791faf1444ba89f764e988)


This code predicts the activity that a participant was performing while wearing a Samsung Galaxy S2 smartphone. The activities are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

The data comes from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.html).

[Here is the link to the page for this data.](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

Here is the R code to download the dataset:

```
download.file("https://spark-public.s3.amazonaws.com/dataanalysis/samsungData.rda", destfile="./samsungData.rda", method="curl")

dateDownloaded <- date() # Date downloanded.
```

The code generates several images that you may or may not find helpful. Please don't be alarmed.
