 ****Wine type prediction: Project Overview****

- Created wine type prediction model based on physicochemical properties of wine
- Visualized relationship between different features
- Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.

****Understanding data features****

There are 1599 samples of red wine and 4898 samples of white wine in the data sets. Each wine sample has the following characteristics :

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (score between 0 and 10)

****Model Building**** 

****Model performance**** 

The Random Forest model far outperformed the other approaches on the test and validation sets.

- DecisionTreeClassifier : 0.73
- RandomForestClassifier: 0.77
- XGBoost: 0.79
- RandomForestClassifier with GridSearchCV: 0.81

****Results(insert images)****

Appendix ****Domain knowledge****
