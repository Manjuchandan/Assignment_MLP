#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random


# In[101]:


X,Y=datasets.make_classification(n_samples=500, n_features=4,n_classes=2,n_redundant=1)


# In[102]:


df=pd.DataFrame(X,columns=['feature1','feature2','feature3','feature4'])
df['label']=Y


# In[103]:


df.head(20)
sns.pairplot(df, hue='label',palette='Set1')


# In[104]:


df.to_csv('random.csv',index=False)


# In[105]:


df1=pd.read_csv('random.csv')


# In[106]:


df1.head(20)


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25,random_state = 1)


# In[108]:


print(y_train)


# In[109]:


class Perceptron:
    def __init__(self,input_size,epochs=100,alpha=0.02):
        self.epochs = epochs
        self.alpha = alpha
        self.input_size = input_size
        weight = [random.random() for i in range(input_size+1)]
        weight[0] = 1.0
        self.weight = weight

    def activation(self,x):
        return int(1 / (1 + np.exp(-x)))
    def predict(self,x):
       
        z = np.dot(self.weight,x)


        a = self.activation(z)
        return a
   
    
    def learn(self,X,d):
        final = []
        for k in range(self.epochs):
            print('Interation',(k+1))
            sum = 0
            for i in range(d.shape[0]):
                x = np.insert(X[i],0,1)
                y = self.predict(x)
                e = (d[i] - y)
                self.weight = self.weight + self.alpha*e*x
                sum = sum + e
            final.append(abs(sum))
            print("Error=",str(sum))
            print('Weights=',self.weight)


# In[110]:


perceptron=Perceptron(input_size=4)
perceptron.learn(X_train,y_train)


# In[111]:


y_pred = []
for i in range(X_train.shape[0]):
    x= np.insert(X_train[i],0,1)
    y_pred.append(perceptron.predict(x))
y_actu = pd.Series(y_train,name = 'Actual')
y_pred = pd.Series(y_pred,name = 'Predicted')

conftrain = pd.crosstab(y_pred,y_actu)
print('Confusion Matrix for Training data\n',conftrain)
accuracy=(conftrain[0][0]+conftrain[1][1])/(conftrain[0][0]+conftrain[1][0]+conftrain[0][1]+conftrain[1][1])
    


# In[112]:


print("Training data Accuracy = " +str(accuracy))


# In[113]:


y_pred = []
for i in range(X_test.shape[0]):
    x= np.insert(X_test[i],0,1)
    y_pred.append(perceptron.predict(x))
y_actu = pd.Series(y_test,name = 'Actual')
y_pred = pd.Series(y_pred,name = 'Predicted')

conftest = pd.crosstab(y_pred,y_actu)
print('Confusion Matrix for Testing data\n',conftest)
accuracy=(conftest[0][0]+conftest[1][1])/(conftest[0][0]+conftest[1][0]+conftest[0][1]+conftest[1][1])
    


# In[114]:


print("Testing data Accuracy = " +str(accuracy))


# In[115]:



y_pred = []
for i in range(X.shape[0]):
    x= np.insert(X[i],0,1)
    y_pred.append(perceptron.predict(x))
y_actu = pd.Series(Y,name = 'Actual')
y_pred = pd.Series(y_pred,name = 'Predicted')

df_confusion = pd.crosstab(y_pred,y_actu)
print('Metrices for whole data\n')
print('\nConfusion Matrix\n\n',df_confusion)
precision=df_confusion[0][0]/(df_confusion[0][0]+df_confusion[0][1])
print('\nPrecision : ',precision)
recall=df_confusion[0][0]/(df_confusion[0][0]+df_confusion[1][0])
print('\nRecall : ',recall)
accuracy=(df_confusion[0][0]+df_confusion[1][1])/(df_confusion[0][0]+df_confusion[1][0]+df_confusion[0][1]+df_confusion[1][1])
print('\nAccuracy : ',accuracy)


# In[83]:


#THANK YOU

