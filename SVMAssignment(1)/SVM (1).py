#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils


# In[8]:


#Generating Random dataset...
X, Y = make_blobs(n_samples=1000, centers=2, 
                  random_state=0, cluster_std=0.40) 


# In[9]:


df=pd.DataFrame(X,columns=['feature1','feature2'])


# In[10]:


df.head(10)


# In[11]:


#Plotting of Data points


# In[12]:


plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring') 


# In[13]:


df.to_csv('random.csv',index=False)


# In[14]:


df1=pd.read_csv('random.csv')


# In[15]:


df1.head(10)


# In[16]:


#Drawing a stright line to separate the 2 sets of data


# In[17]:


#Drawing a stright line to separate the 2 sets of data
xfit = np.linspace(-2,4.5)
plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap='spring')

plt.plot([0.6],[2.1] , 'x' , color='red', markeredgewidth=2 , markersize=10)

for m,b in [(1,0.65) , (0.5 , 1.6) , (-0.2 , 2.9)]:

 plt.plot(xfit , m * xfit + b , '-k')

plt.xlim(-2, 4.5);


# In[18]:


xfit = np.linspace(-2, 4.5) 


# In[19]:


plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap='spring')
  
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
    yfit = m * xfit + b 
    plt.plot(xfit, yfit, '-k') 
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',  
    color='#AAAAAA', alpha=0.4) 
  
plt.xlim(-2, 4.5); 
plt.show() 


# In[20]:


x=pd.read_csv('random.csv')


# In[21]:


a=np.array(x)


# In[22]:


y=a[:,1]


# In[23]:


x = np.column_stack((x.feature1,x.feature2)) 
x.shape # 569 samples and 2 features 
  
print (x),(y) 


# In[24]:


from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))


# In[25]:


#Fitting a SVM Model

clf = SVC(kernel='linear') 
  
# fitting 
clf.fit(x, training_scores_encoded) 


# In[26]:


#SVM Decision Boundary Function


# In[27]:


def SVM_Decision_Boundary(clf, ax=None, plot_support=True):
    
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = clf.decision_function(xy).reshape(X.shape)
    

    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0],
                   clf.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# In[28]:


#plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap='spring')
#SVM_Decision_Boundary(clf);


# In[29]:


#Support Vectors Points


# In[30]:


clf.support_vectors_


# In[31]:


# def plot_svm(N=10, ax=None):
#     X, y = make_blobs(n_samples=1000, centers=2,
#                       random_state=0, cluster_std=0.40)
#     X = X[:N]
#     y = y[:N]
#     clf = SVC(kernel='linear', C=1E10)
#     clf.fit(x, training_scores_encoded) 
    
#     ax = ax or plt.gca()
#     ax.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
#     ax.set_xlim(-1, 4)
#     ax.set_ylim(-1, 6)
#     SVM_Decision_Boundary(clf, ax)

# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, N in zip(ax, [1000, 120]):
#     plot_svm(N, axi)
#     axi.set_title('N = {0}'.format(N))


# In[32]:


# from ipywidgets import interact, fixed
# interact(plot_svm, N=[10, 200], ax=fixed(None));


# In[33]:


df=pd.DataFrame(X,Y)


# In[34]:


df1=df


# In[36]:


#Dropping of Rows which are not Support Vectors and Store it into df2...

df2=df1.drop(df1.index == 'clf.support_vectors_')


# In[37]:


#Storing the Support Vectors into df3


# In[38]:


df3=df.drop(df.index != 'clf.support_vectors_')


# In[39]:


# NO of Rows which are not Support Vectors
# 500 Rows

df2.count()


# In[40]:


# NO of Support Vectors points
# 500

df3.count()


# In[41]:


# Fitting a SVM model after removing some data points other than Support Vectors


# In[42]:


clf = SVC(kernel='linear') 
  
# fitting 
clf.fit(x, training_scores_encoded) 


# In[43]:


# Below cell shows removing of data points other than Support Vectors does not affect Decision Boundary..


# In[44]:


clf.support_vectors_


# In[50]:


# Thank You

