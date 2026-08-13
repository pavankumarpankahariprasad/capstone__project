Part A — Profiling, Cleaning and Data Story
1. Loading the Dataset

The Titanic dataset is loaded using Pandas:

import pandas as pd
import numpy as np

df = pd.read_csv("Titanic-Dataset.csv")

The EDA notebook then checks the shape, information, and descriptive statistics:

print(df.shape)
print(df.info())
print(df.describe())

The project also calculates the percentage of missing values:

df.isna().sum() * 100 / df.shape[0]

These steps provide an initial understanding of the dataset, its dimensions, data types, numerical statistics, and missing values.

2. Missing-Value Handling

The project applies different strategies depending on the amount and importance of missing data.

Embarked

The Embarked column contains a small percentage of missing values.

The missing rows are removed:

df = df.dropna(subset=["Embarked"])

This approach is reasonable because only a small number of observations are affected.

Cabin

The Cabin column contains a very high percentage of missing values.

Instead of attempting unreliable imputation, the column is removed:

df = df.drop(columns=["Cabin"])

The reasoning is that the high proportion of missing values makes direct imputation unreliable.

Age

The Age column has a moderate amount of missing data.

The current notebook fills missing ages using the mode:

df["Age"] = df["Age"].fillna(df["Age"].mode()[0])

The resulting column is then converted to integer:

df["Age"] = df["Age"].astype(int)
Missing-Value Strategy Summary
Column	Strategy
Embarked	Drop rows with missing values
Cabin	Drop entire column
Age	Mode imputation
Other columns	Retained according to availability
3. Univariate Analysis

Univariate analysis examines individual variables separately.

The project analyzes:

Age
Fare
Age Histogram

A histogram is used to visualize the distribution of passenger ages:

plt.hist(
    df["Age"],
    edgecolor="black",
    bins=20
)

The histogram helps understand the frequency distribution of passenger ages.

Fare Histogram

The Fare distribution is also visualized:

plt.hist(
    df["Fare"],
    edgecolor="black"
)

Fare generally contains a number of relatively high values compared with the majority of passengers.

Box Plots

Box plots are created for both Age and Fare:

plt.boxplot(df["Age"])

and:

plt.boxplot(df["Fare"])

Box plots help identify potential outliers.

4. IQR Outlier Detection

The project uses the Interquartile Range (IQR) method.

The formula is:

IQR = Q3 - Q1

Lower bound:

Q1 - 1.5 × IQR

Upper bound:

Q3 + 1.5 × IQR

The notebook calculates outliers for numerical columns using:

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df[col] < lower_limit) |
    (df[col] > upper_limit)
]

The number of outliers is then printed for each numerical column.

5. Survival Rate Analysis

The project investigates survival rates based on:

Sex
Passenger class
Sex and passenger class together
Survival Rate by Sex

The survival rate for males and females is calculated:

male = df[df["Sex"] == "male"]["Survived"].mean() * 100

female = df[df["Sex"] == "female"]["Survived"].mean() * 100

This analysis helps determine the relationship between passenger sex and survival.

6. Survival Rate by Passenger Class

Survival rates are calculated separately for:

First class
Second class
Third class

Example:

class1 = df[df["Pclass"] == 1]["Survived"].mean() * 100

class2 = df[df["Pclass"] == 2]["Survived"].mean() * 100

class3 = df[df["Pclass"] == 3]["Survived"].mean() * 100

This allows comparison of survival probability across passenger classes.

7. Survival Rate by Sex and Passenger Class

The project also combines Sex and Pclass.

Boolean masking using & is used:

female_class1 = df[
    (df["Sex"] == "female") &
    (df["Pclass"] == 1)
]["Survived"].mean() * 100

Similar calculations are performed for all combinations of:

Female + Class 1
Female + Class 2
Female + Class 3
Male + Class 1
Male + Class 2
Male + Class 3

This provides a more detailed picture of which passenger groups had higher survival rates.

8. Correlation Analysis

The EDA notebook calculates a correlation matrix for numerical variables.

The correlation matrix is visualized using Seaborn:

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

Correlation values range between:

-1 and +1

where:

+1 indicates a strong positive relationship.
-1 indicates a strong negative relationship.
0 indicates little or no linear relationship.

The project also ranks correlation pairs based on their absolute correlation values to identify the strongest relationships.

9. Standardization of Age and Fare

As an exploratory analysis step, Age and Fare are standardized using StandardScaler.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = df.copy()

df_scaled[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

The standardization formula is:

z = (x - mean) / standard deviation

Before and after statistics are printed:

print("Before Scaling")
print(df[["Age", "Fare"]].agg(["mean", "std"]))

print("\nAfter Scaling")
print(df_scaled[["Age", "Fare"]].agg(["mean", "std"]))

The transformed variables should have approximately:

Mean ≈ 0
Standard deviation ≈ 1

Histograms are also used to visually compare the variables before and after scaling.

Part B — Predictive Modeling
10. Feature Selection

For the classification task, the target is:

y = df["Survived"]

The feature matrix is:

x = df.drop(columns=["Survived"])

The current modeling notebook removes several columns:

df = df.drop(
    columns=[
        "PassengerId",
        "Ticket",
        "SibSp",
        "Name"
    ]
)

Categorical variables are converted using one-hot encoding:

df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)
11. Train/Test Split

The data is divided into training and testing sets:

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

The test set contains 20% of the data.

The modeling requirement recommends a stratified split so that the proportion of survived and not-survived observations remains similar in the training and testing datasets.

For a final submission, the split should therefore use:

stratify=y
12. Feature Scaling

The current modeling notebook uses StandardScaler:

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

Importantly, the scaler is fitted on the training data and then used to transform the test data.

This avoids using test-set statistics during training.

13. Classification Models

Three classification algorithms are used.

Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(
    x_train_scaled,
    y_train
)
Decision Tree
from sklearn.tree import DecisionTreeClassifier

model1 = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model1.fit(
    x_train_scaled,
    y_train
)
Random Forest
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=42
)

model2.fit(
    x_train_scaled,
    y_train
)
14. Decision Tree Visualization

The Decision Tree is visualized using plot_tree:

from sklearn.tree import plot_tree

plot_tree(
    model1,
    feature_names=x_train.columns.tolist(),
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=3
)

The visualization makes the tree's decision rules easier to understand.

15. Classification Evaluation

All three classifiers are evaluated using:

Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC-AUC

The metrics are calculated using:

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

The results are stored in a comparison DataFrame.

16. Classification Results

The current notebook reports the following results for the Logistic Regression imbalance comparison:

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Baseline Logistic Regression	0.7865	0.7013	0.7826	0.7397	0.8506
Oversampling Logistic Regression	0.7865	0.7013	0.7826	0.7397	0.8506
Balanced Logistic Regression	0.7753	0.6706	0.8261	0.7403	0.8490
Interpretation

The baseline model achieved the highest accuracy, precision, and ROC-AUC among these three variants. The balanced model achieved the highest recall, meaning it identified more actual survivors, but its precision and accuracy were lower. The oversampling result is identical to the baseline in the current notebook because the SMOTE-resampled training data is created but the oversampling model is fitted on x_train_scaled rather than on x_train_smote.

17. Imbalance Handling

Three approaches are considered:

Baseline

No class imbalance handling:

baseline = LogisticRegression(
    max_iter=1000
)
Class Weight

The balanced class-weight approach uses:

balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
SMOTE

SMOTE is created using:

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(
    x_train_scaled,
    y_train
)

For a correct SMOTE comparison, the model must be trained using:

oversampling.fit(
    x_train_smote,
    y_train_smote
)

rather than the original x_train_scaled.

SMOTE must be applied only to the training data to prevent data leakage.

18. Random Forest Hyperparameter Tuning

GridSearchCV is used to search for the best Random Forest parameters.

The search includes:

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "max_features": ["sqrt", "log2"]
}

The Random Forest estimator is created with OOB scoring enabled:

rf = RandomForestClassifier(
    oob_score=True,
    random_state=42
)

Grid search is then performed:

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

The best parameters are obtained using:

grid_search.best_params_

The best cross-validation ROC-AUC is obtained using:

grid_search.best_score_

The OOB score is obtained from:

best_rf.oob_score_
19. Regression Side Task

A multivariate linear regression model is used to predict:

Fare

The target is:

y = df["Fare"]

The other available features are used as predictors:

x = df.drop(columns=["Fare"])

The data is divided into training and testing sets.

The features are standardized using StandardScaler.

A Linear Regression model is then trained:

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear_model = linear.fit(
    x_train_scale,
    y_train
)
20. Regression Metrics

The regression model is evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R²
Adjusted R²

The metrics are calculated using:

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
MAE

MAE measures the average absolute difference between the actual and predicted fare.

MAE = average(|actual - predicted|)
RMSE

RMSE gives greater weight to larger errors.

RMSE = sqrt(mean((actual - predicted)²))
R²

R² measures the proportion of variance in Fare explained by the predictors.

Adjusted R²

Adjusted R² accounts for the number of predictors in the regression model.

The project calculates it using:

adj_r2 = 1 - (
    (1 - r2) *
    (n - 1) /
    (n - k - 1)
)
21. Model Comparison

Classification and regression are different machine-learning tasks.

Therefore, their metrics should not be interpreted as being on one common performance scale.

Classification Metrics

The classifiers are compared using:

Model	Accuracy	Precision	Recall	F1	AUC
Baseline Logistic Regression	0.7865	0.7013	0.7826	0.7397	0.8506
Oversampling Logistic Regression	0.7865	0.7013	0.7826	0.7397	0.8506
Balanced Logistic Regression	0.7753	0.6706	0.8261	0.7403	0.8490
Regression Metrics

The regression model is evaluated separately using:

Model	MAE	RMSE	R²	Adjusted R²
Multivariate Linear Regression	Calculated in notebook	Calculated in notebook	Calculated in notebook	Calculated in notebook

Classification and regression metrics are deliberately presented as separate metric groups because they measure different types of prediction problems.

22. Model Recommendation

Based on the currently reported classification results, Logistic Regression with the baseline configuration is a strong overall choice because it provides the highest accuracy of approximately 78.65%, precision of approximately 70.13%, and ROC-AUC of approximately 0.851 among the reported imbalance variants. Its recall is approximately 78.26%, which means it identifies a substantial proportion of actual survivors. The balanced model has slightly higher F1 and substantially higher recall, so it would be preferable if missing actual survivors were considered more costly than false-positive predictions. Overall, the baseline Logistic Regression provides the best balance of accuracy, precision, recall, F1, and AUC in the currently reported results.

23. Saving the Model

The tuned Random Forest model is saved using Joblib:

import joblib

joblib.dump(
    best_rf,
    "RandomForest_CV_model.pkl"
)

The regression model is also saved:

joblib.dump(
    linear_model,
    "linear_regression.pkl"
)
24. Important Model-Persistence Note

For a production-ready submission, the complete preprocessing pipeline should be saved together with the final classifier.

The recommended structure is:

Raw Input
    ↓
Imputer
    ↓
Encoder
    ↓
Scaler
    ↓
Classifier

This can be implemented using a scikit-learn Pipeline and ColumnTransformer.

The final artifact should be saved as:

joblib.dump(
    full_pipeline,
    "titanic_full_pipeline.pkl"
)

This is preferable to saving only the classifier because the saved object can receive raw, unprocessed input and perform the required preprocessing automatically.

25. Loading the Saved Pipeline

The saved pipeline can later be loaded using:

import joblib

loaded_pipeline = joblib.load(
    "titanic_full_pipeline.pkl"
)

A prediction can then be made directly from raw input:

prediction = loaded_pipeline.predict(new_data)

print(prediction)

This confirms that the preprocessing and model are stored together.
