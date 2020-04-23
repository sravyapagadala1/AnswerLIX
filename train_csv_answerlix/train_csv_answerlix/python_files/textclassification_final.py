
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from textblob.classifiers import NaiveBayesClassifier as NBC


# In[25]:


df = pd.read_csv("M:\\GMU@Sem3\\DAEN 690\\Datasets\\possible_question_types_final.csv",header = None)
test =pd.read_csv('C:\\Users\\mohan\\Desktop\\Final Project\\Inputs_final\\new_testttt.csv',header = None)
df.head()


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models = [LinearSVC(),
    LogisticRegression(random_state=0)]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X_test_tfidf,y_test, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[23]:


cv_df.groupby('model_name').accuracy.mean()


# In[24]:


d = df.values.tolist()


# In[14]:


x_train ,x_test = train_test_split(d,test_size=0.7) 


# In[27]:


mod = NBC(x_train) 


# In[29]:


print("Naive Bayes Classifier Accuracy is",mod.accuracy(x_test))


# In[30]:


target = []
for i in test[0]:
    target.append(mod.classify(i))


# In[31]:


target


# In[32]:


actual = test[1]
predicted = target
count = 0 
total = len(actual)
for i in range(0,total):
    if (actual[i] == predicted[i]):
        count = count+1
print('Accuracy :',count/total*100)

