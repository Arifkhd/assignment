#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['figure.dpi'] = 500
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


#Data Exploration:
#Load the dataset and perform exploratory data analysis (EDA).
df1 = pd.read_csv(r"C:\Users\Dell\Downloads\Logistic Regression\Logistic Regression\Titanic_train.csv")
df2 = pd.read_csv(r"C:\Users\Dell\Downloads\Logistic Regression\Logistic Regression\Titanic_test.csv")


# In[32]:


df1


# In[33]:


df2


# In[34]:


if 'Survived' not in df3.columns:
    df2['Survived'] = None


# In[35]:


df3 = pd.concat([df1,df2,],ignore_index=True)


# In[36]:


df3


# In[38]:


#Examine the features, their types, and summary statistics.
df3.describe()


# In[39]:


df3.dtypes


# In[40]:


df3.info


# In[41]:


#Create visualizations such as histograms, box plots, or pair plots to visualize
#the distributions and relationships between features.
df3.hist()


# In[42]:


df3.boxplot()


# In[43]:


sns.pairplot(df3)


# In[44]:


df3.corr()


# In[45]:


sns.heatmap(df3.corr(),annot=True,cmap='coolwarm')


# In[50]:


#Data Preprocessing:
#Handle missing values (e.g., imputation).
df3.isnull().sum()


# In[49]:


for i in df3.isnull().sum():
    print((i/891)*100,'%')


# In[48]:


df3 = df3.dropna()


# In[51]:


#Encode categorical variables:
#we have to convert categorial data into numerical since machine learning doesn't understand categorial value
df3['Sex'] = df3['Sex'].map({'male': 0, 'female': 1}) #here map() will replace each value 
df3['Embarked'] = df3['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[53]:


df3


# In[54]:


data = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = df3[data]
y = df3['Survived'].values
y.shape


# In[55]:


x


# In[56]:


#Split the data
train_test_split(x,y,train_size=0.8)


# In[57]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# In[58]:


x_train


# In[59]:


y_train


# In[60]:


x_test


# In[61]:


y_test


# In[62]:


#creating the model
#we already import the library above
LR = LogisticRegression()
LR.fit(x_train,y_train)


# In[63]:


#evaluate the model
y_pred = LR.predict(x_test)
y_pred
y_pred_proba = LR.predict_proba(x_test)[:, 1]
y_pred_proba


# In[66]:


#we a have already import all the required library above
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)


# In[67]:


print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")


# In[68]:


#calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")


# In[69]:


#Visualize the ROC curve
plt.figure(figsize=(8, 6)) #size of the graph
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')#fpr represents false +ve rate and tpr represents true +ve rate
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0]) #xlim is used to set the limits of the x-axis of a plot
plt.ylim([0.0, 1.05]) #ylim is used to set the limit of the y-axis of a plot
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')


# In[71]:


coefficients = pd.DataFrame(LR.coef_.T, index=data, columns=['Coefficient'])
print(coefficients)


# In[ ]:




