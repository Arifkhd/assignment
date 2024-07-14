#!/usr/bin/env python
# coding: utf-8

# In[107]:


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


# In[108]:


#Data Exploration:
#Load the dataset and perform exploratory data analysis (EDA).
df = pd.read_csv(r"C:\Users\Dell\Downloads\Logistic Regression\Logistic Regression\Titanic_train.csv")
df


# In[109]:


#Examine the features, their types, and summary statistics.
df.describe()


# In[110]:


df.dtypes


# In[111]:


df.info


# In[112]:


#Create visualizations such as histograms, box plots, or pair plots to visualize
#the distributions and relationships between features.
df.hist()


# In[113]:


df.boxplot()


# In[114]:


sns.pairplot(df)


# In[115]:


df.corr()


# In[116]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# In[117]:


#Data Preprocessing:
#Handle missing values (e.g., imputation).
df.isnull().sum()


# In[118]:


for i in df.isnull().sum():
    print((i/891)*100,'%')


# In[119]:


df['Age'].median()


# In[120]:


df['Age'].fillna(df['Age'].median(),inplace=True)


# In[121]:


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[122]:


df.isnull().sum()


# In[123]:


#Encode categorical variables:
#we have to convert categorial data into numerical since machine learning doesn't understand categorial value
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) #here map() will replace each value 
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[124]:


df


# In[125]:


data = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x = df[data]
y = df['Survived'].values
y.shape


# In[126]:


x


# In[127]:


#Split the data
train_test_split(x,y,train_size=0.8)


# In[128]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# In[129]:


x_train


# In[130]:


y_train


# In[131]:


x_test


# In[132]:


y_test


# In[133]:


#creating the model
#we already import the library above
LR = LogisticRegression()
LR.fit(x_train,y_train)


# In[134]:


#evaluate the model
y_pred = LR.predict(x_test)
y_pred
y_pred_proba = LR.predict_proba(x_test)[:, 1]
y_pred_proba


# In[135]:


#we a have already import all the required library above
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)


# In[136]:


print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")


# In[137]:


#calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")


# In[138]:


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


# In[139]:


coefficients = pd.DataFrame(LR.coef_.T, index=data, columns=['Coefficient'])
print(coefficients)


# In[ ]:




