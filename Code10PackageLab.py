#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[6]:


# Pull in our dataset
data = {

    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],

    'BloodPressure': [120, 122, 126, 128, 130, 133, 135, 138, 142, 145, 150, 155, 160, 165, 170, 175]

}


# In[26]:


df = pd.DataFrame(data)

df.descriptive = df.describe()

print(df.descriptive)


# In[17]:


# Scatter plot of Age vs Blood Pressure
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['BloodPressure'], color='cyan')
plt.xlabel('Age')
plt.ylabel('BloodPressure')
plt.title('Age vs BloodPressure')
plt.show


# In[21]:


# Linear Regression
x = df[['Age']]
y = df['BloodPressure']
regression = LinearRegression().fit(x,y)


# In[22]:


plt.plot(x, regression.predict(x))


# In[23]:


plt.plot(x, regression.predict(x), label = 'Regression Line', color = 'pink')
plt.scatter(df['Age'], df['BloodPressure'], color='cyan')
plt.show


# In[25]:


slope = regression.coef_[0]
intercept = regression.intercept_
print(f"Regression model has slope of {slope} and intercept of {intercept}.")


# In[29]:


# Predictions
new_ages = [30, 40, 50, 60]
df_ages = pd.DataFrame({'Age': new_ages})
predicted_blood_pressures = regression.predict(df_ages)


# In[30]:


for age, bp in zip(new_ages, predicted_blood_pressures):
    print(f"Predicted Blood Pressure at Age {age} is {bp:.2f}.")


# In[ ]:




