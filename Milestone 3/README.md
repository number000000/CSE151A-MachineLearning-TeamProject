<a target="_blank" href="https://colab.research.google.com/drive/1kSHfct3UeFsFB2DUTZJE6G6X70RQnjvS">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Pre-processing
We preprocess the data and build our first keras model using Sequential. 

## Fitting
Accuracy, Precision, Recall, and F1-score are consistently high on the test dataset (around 93%), indicating that the model performs well on unseen data.

Loss metrics show that the validation loss (with values ranging from about 0.20 to 0.24) is reasonably close to the training loss (which decreases to about 0.23 by the end of training), and the test loss is also low (0.1883).

We can see the training, validating, and testing metrics perform good and similar to each other which show that our model is just right. No overfitting or underfitting happen.


## Next models to try
1. SVM: SVM is a good model for classication, and we are trying to classify rice base on features, which is appropriate. 
2. Naive Bayes: Naive Bayes is a good model for classication task, and we are trying to classify rice base on features, which is appropriate.

## Conclusion
The model currently perform well, with above 90% accuracy on testing data. However, there still some room to improve. We can tune our hyperparameter to try to achieve better metrics.
