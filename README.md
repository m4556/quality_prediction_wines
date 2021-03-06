  #  ****Wine quality prediction: Project Overview****

- Predicted the quality of a wine based on physicochemical properties of wine.
- Visualized relationship between different features.
- Optimized different tree based models using GridsearchCV to reach the best model.

 # ****Data features****

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

 #  ****Model performance**** 

The Random Forest model far outperformed the other approaches on the test and validation sets.

- DecisionTreeClassifierÂ : 0.73
- RandomForestClassifier: 0.77
- XGBoost: 0.79
- RandomForestClassifier with GridSearchCV: 0.81
