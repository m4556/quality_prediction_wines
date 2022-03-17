#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


# ## Predicting wine quality
# 

# ### Feature selection, train test set

# In[ ]:


wines = pd.read_csv('data/wines.csv')
wine_features = wines.iloc[:,:-3]   
label_names = ['low', 'medium', 'high']
feature_names = wine_features.columns

train_X, test_X, train_y, test_y = train_test_split(wine_features, wines['quality_label'],test_size=0.3, random_state=42) 

print(Counter(train_y),Counter(test_y)) 


# ### Feature scaling

# In[ ]:


scaler = StandardScaler().fit(train_X)

train_X_scaled = scaler.transform(train_X)
test_X_scaled = scaler.transform(test_X)


# ### Model building, prediction and evaluation using Decision Trees

# In[ ]:


model_dt = DecisionTreeClassifier()
model_dt.fit(train_X_scaled, train_y)

predictions = model_dt.predict(test_X_scaled)

meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=predictions, classes=label_names)


# ### Feature importance for Decision Trees

# In[ ]:


plt.title('Feature Importances for Decision Tree')

feat_importances = pd.Series(model_dt.feature_importances_, index=feature_names)
f= feat_importances.nlargest(13).plot(kind='barh',color='y',alpha=0.5)


# ### Model building, prediction and evaluation using Random Forests

# In[ ]:


model_rf = RandomForestClassifier()
model_rf.fit(train_X_scaled, train_y)

predictions = model_rf.predict(test_X_scaled)

meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=predictions, 
                                      classes=label_names)


# ### Model tuning using GridSearchCV

# In[ ]:


parameters_grid = {'n_estimators': [100, 200, 300, 500],'max_features': ['auto', None, 'log2']}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), parameters_grid, cv=5, scoring='accuracy')

grid_search_rf.fit(train_X_scaled, train_y)

print(grid_search_rf.best_params_)

results = grid_search_rf.cv_results_
for param, score_mean, score_sd in zip(results['params'], results['mean_test_score'], results['std_test_score']):
    print(param, round(score_mean, 4), round(score_sd, 4))


# ### Random Forests with tuned hyperparameters

# In[ ]:


model_rf = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=42)
model_rf.fit(train_X_scaled, train_y)

predictions = model_rf.predict(test_X_scaled)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=predictions, 
                                      classes=label_names)


# ### Model building, prediction and evaluation using ensemble methods: XBoost

# In[ ]:


model_xgb = xgb.XGBClassifier(seed=42, num_class=14,eval_metric='mlogloss')
model_xgb.fit(train_X_scaled, train_y)

predictions = model_xgb.predict(test_X_scaled)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=predictions, 
                                      classes=label_names)


# ### Model tuning using GridSearchCV

# In[ ]:


parameters_grid = {'n_estimators': [100, 200, 300, 500],'max_depth': [5, 10, 15],'learning_rate': [0.3, 0.5]}
grid_search_xgb = GridSearchCV(xgb.XGBClassifier(tree_method='exact', seed=42, num_class=14,eval_metric='mlogloss', use_label_encoder=False), parameters_grid, cv=5, scoring='accuracy')

grid_search_xgb.fit(train_X_scaled, train_y)

print(grid_search_xgb.best_params_)

results = grid_search_xgb.cv_results_


# ### XBoost with tuned hyperparameters

# In[ ]:


model_xgb = xgb.XGBClassifier(seed=42,num_class=14, max_depth=10, learning_rate=0.3, eval_metric='mlogloss')
model_xgb.fit(train_X_scaled, train_y)

predictions = model_xgb.predict(test_X_scaled)
meu.display_model_performance_metrics(true_labels=test_y, predicted_labels=predictions, 
                                      classes=label_names)

