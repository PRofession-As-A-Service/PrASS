#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[20]:


# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


# Load the data
data = pd.read_csv('D:/Ml Project/PrAAS - Profession as a Service - Aspire Infolabs Hackathon/TraitsVsRole.csv')  # Replace 'your_data.csv' with the actual file name


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[13]:


data.describe()


# In[14]:


# Split the data into features and target
X = data.drop(['Role', 'Name'], axis=1)
y = data['Role']


# In[15]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:



# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[17]:


# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[18]:



# Make predictions
y_pred = model.predict(X_test_scaled)


# In[19]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# # Random Forest

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[22]:


# Load the data
data = pd.read_csv('D:/Ml Project/PrAAS - Profession as a Service - Aspire Infolabs Hackathon/TraitsVsRole.csv')  # Replace 'your_data.csv' with the actual file name


# In[23]:


# Split the data into features and target
X = data.drop(['Role', 'Name'], axis=1)
y = data['Role']


# In[24]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Create and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)


# In[26]:


# Make predictions
y_pred_rf = model_rf.predict(X_test)



# In[27]:


# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')


# # SVM

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[30]:


# Load the data
data = pd.read_csv('D:/Ml Project/PrAAS - Profession as a Service - Aspire Infolabs Hackathon/TraitsVsRole.csv')  # Replace 'your_data.csv' with the actual file name


# In[31]:


# Split the data into features and target
X = data.drop(['Role', 'Name'], axis=1)
y = data['Role']


# In[32]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


# Create and train the SVM model
model_svm = SVC(kernel='linear', C=1.0, random_state=42)
model_svm.fit(X_train, y_train)



# In[34]:


# Make predictions
y_pred_svm = model_svm.predict(X_test)


# In[35]:


# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')


# # Gradient Boosting

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[38]:


# Load the data
data = pd.read_csv('D:/Ml Project/PrAAS - Profession as a Service - Aspire Infolabs Hackathon/TraitsVsRole.csv')  # Replace 'your_data.csv' with the actual file name


# In[39]:


# Split the data into features and target
X = data.drop(['Role', 'Name'], axis=1)
y = data['Role']


# In[40]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[41]:


# Create and train the Gradient Boosting model
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model_gb.fit(X_train, y_train)


# In[42]:


# Make predictions
y_pred_gb = model_gb.predict(X_test)


# In[43]:


# Calculate accuracy
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f'Gradient Boosting Accuracy: {accuracy_gb:.2f}')


# In[ ]:




