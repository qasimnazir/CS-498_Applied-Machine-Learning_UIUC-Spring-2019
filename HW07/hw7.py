"""
CS 498: Applied Machine Learning (Spring 2019)
Homework 7: Text Bag-of-Words Search and Classification
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn import metrics
"""
Part 1: Preprocessing with Bag-of-Words
"""
# 1. import data
data = pd.read_csv('yelp_2k.csv')

# extract features and labels data
X = data['text'].values
y = data['stars'].values
y = binarize(y.reshape(len(X),1),threshold=1.0).reshape(len(X),)

# 2. convert text to lowercase
X = [x.lower() for x in X]

# 3. Bag-of-words Analysis and Repreprocessing
# convert X to Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=True)
X1 = vectorizer.fit_transform(X).toarray()
all_words = np.array(vectorizer.get_feature_names())
frequency = np.sum(X1,axis=0)
indexbyfreq = np.argsort(-frequency)

# word frequency plot
plt.figure(1)
plt.plot(-np.sort(-frequency)[0:5000],'.',markersize=2)
plt.title('Word Frequency (Original Data)')
plt.xlabel('Word Rank')
plt.ylabel('Word Count')

# select stop words
freq_thresh = 800
No_stopwords = len(all_words[frequency>freq_thresh])
stopwords = all_words[indexbyfreq[:No_stopwords]]

# Re-Process Data
vectorizer2 = CountVectorizer(stop_words=list(stopwords),max_df=0.9,min_df=5)
X2 = vectorizer2.fit_transform(X).toarray()
frequency2 = np.sum(X2,axis=0)
bag_of_words = vectorizer2.get_feature_names()

# again word frequency plot
plt.figure(2)
plt.plot(-np.sort(-frequency2),'.',markersize=2)
plt.title('Word Frequency (Reprocessed Data)')
plt.xlabel('Word Rank')
plt.ylabel('Word Count')

"""
Part 2: Text-Retrieval
"""
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

target=['horrible custome service']
target=vectorizer2.transform(target)
sparse=vectorizer2.transform(X)
score=cosine_similarity(target,sparse)
sorted_index= np.fliplr((np.argsort(score)))
#setting so it can print out everything
np.set_printoptions(threshold=np.inf)
max_5=[sorted_index[0,i] for i in range(5)]
for j,i in enumerate(max_5):
    print("Review",j+1,":",X[i][:200])
    print("Score:",score[0,i])
    
"""
Part 3: Classification with Logistic Regression
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# train classifier and determine accuracies
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.1, random_state = 0)
classifier = LogisticRegression(random_state = 0).fit(X_train, y_train)
acc_train = classifier.score(X_train,y_train)
acc_test  = classifier.score(X_test,y_test)
print("Train Set Accuracy:",acc_train)
print("Test Set Accuracy:",acc_test)

# Plot Histogram of Predicted Scores
pred_score_train = classifier.predict_proba(X_train)[:,1]
plt.figure(3)
plt.hist(pred_score_train[y_train==0],bins=80,label='1 Star',color='b')
plt.hist(pred_score_train[y_train==1],bins=80,label='5 Star',color='g')
plt.title('Histogram of Predicted Scores')
plt.ylabel('Count of Predictions in Bucket')
plt.xlabel('Predicted Score')
plt.legend()
#plt.show()

# Chnage the Theshold
new_threshold = 0.55
# for train data
pred_score_train = classifier.predict_proba(X_train)[:,1]
y_pred_train = np.ones(len(pred_score_train))
y_pred_train[pred_score_train < new_threshold] = 0
acc_train_new = accuracy_score(y_pred_train,y_train)
# for test data
pred_score_test = classifier.predict_proba(X_test)[:,1]
y_pred_test = np.ones(len(pred_score_test))
y_pred_test[pred_score_test < new_threshold] = 0
acc_test_new = accuracy_score(y_pred_test,y_test)
print("Train Set Accuracy:",acc_train_new)
print("Test Set Accuracy:",acc_test_new)

from sklearn.metrics import roc_curve, auc, roc_auc_score
plt.figure(4)
fpr, tpr, threshold = roc_curve(y_test, pred_score_test)
auc = roc_auc_score(y_test, pred_score_test)
plt.plot(fpr,tpr,color='darkorange',label="ROC Curve (area="+str(auc)+")")
plt.plot(threshold,threshold,'b.-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve for Test Data')
plt.xlabel("False Postitive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# for i in range(len(fpr)):
# 	print(fpr[i],tpr[i],threshold[i])

# candidate_threeshold=np.arange(0.0,1.04,0.04)
# false_positive=np.zeros(len(candidate_threeshold))
# true_positive=np.zeros(len(candidate_threeshold))

# for i,threshold in enumerate(candidate_threeshold):
# 	y_pred = np.ones(len(pred_score_test))
# 	y_pred[pred_score_test<threshold]=5
# 	cm = confusion_matrix(y_test, y_pred)
# 	true_positive[i]=cm[1,1]/(cm[1,0]+cm[1,1])
# 	false_positive[i]=cm[0,1]/(cm[0,1]+cm[0,0])

# print(true_positive)
# print(false_positive)
# plt.figure()
# plt.plot(false_positive,true_positive)
# plt.title('ROC Curve for Test Data')
# plt.xlabel("False Postitive Rate")
# plt.ylabel("True Positive Rate")
# plt.show()
# at about 0.12 false positive rate the true posittive rate could be maximized 

