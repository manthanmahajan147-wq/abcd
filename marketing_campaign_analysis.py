# Advanced Analytics for Decision-Making - End Term Project
# Marketing Campaign Customer Analytics
# Dataset: marketing_campaign_DS14.csv
#
# Purpose:
# 1. Data cleaning and preparation
# 2. EDA
# 3. Linear Regression
# 4. Logistic Regression
# 5. Cluster Analysis
# 6. Decision Tree
# 7. Random Forest
# 8. Ridge/Lasso robustness check
#
# Install if required:
# pip install pandas numpy matplotlib scikit-learn python-docx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("marketing_campaign_DS14.csv", sep=None, engine="python")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))

# ============================================================
# 2. DATA PREPARATION
# ============================================================
# Convert customer date from text to datetime.
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"], dayfirst=True, errors="coerce"
)

# Missing-value check
print("Missing values:")
print(df.isna().sum())

# Duplicate check
print("Duplicate rows:", df.duplicated().sum())

# Identify categorical and numerical variables
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

print("Categorical:", categorical_cols)
print("Numerical:", numeric_cols)

# IQR outlier detection for continuous variables.
continuous_cols = [
    "Income","Recency","MntWines","MntFruits","MntMeatProducts",
    "MntFishProducts","MntSweetProducts","MntGoldProds",
    "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
    "NumStorePurchases","NumWebVisitsMonth","Total_Spending",
    "Total_Purchases","Avg_Spend_Per_Purchase","Customer_Tenure_Months",
    "Digital_Engagement_Score","Estimated_CLV","Churn_Risk_Score",
    "Loyalty_Score","Customer_Profitability","Satisfaction_Score"
]

for c in continuous_cols:
    q1 = df[c].quantile(.25)
    q3 = df[c].quantile(.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = ((df[c] < low) | (df[c] > high)).sum()
    print(c, "outliers:", count)

# Cap continuous predictor outliers for modeling.
# Original data is retained for descriptive analysis.
model_df = df.copy()
for c in continuous_cols:
    q1 = model_df[c].quantile(.25)
    q3 = model_df[c].quantile(.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    model_df[c] = model_df[c].clip(low, high)

# ============================================================
# 3. EDA
# ============================================================
print("Response rate:", df["Response"].mean())
print("Average income:", df["Income"].mean())
print("Average spending:", df["Total_Spending"].mean())

print(df["Education"].value_counts())
print(df["Marital_Status"].value_counts())
print(df["Preferred_Channel"].value_counts())
print(df["Customer_Segment"].value_counts())

# Descriptive statistics
print(df[[
    "Income","Recency","Total_Spending","Total_Purchases",
    "Digital_Engagement_Score","Loyalty_Score","Satisfaction_Score"
]].describe())

# Spending distribution
plt.figure(figsize=(8,5))
plt.hist(df["Total_Spending"], bins=25)
plt.title("Distribution of Total Customer Spending")
plt.xlabel("Total Spending")
plt.ylabel("Customers")
plt.show()

# Response by customer segment
response_segment = df.groupby("Customer_Segment")["Response"].mean() * 100
response_segment.sort_values(ascending=False).plot(kind="bar")
plt.title("Campaign Response Rate by Customer Segment")
plt.ylabel("Response Rate (%)")
plt.show()

# Income vs spending
plt.figure(figsize=(8,5))
plt.scatter(df["Income"], df["Total_Spending"], alpha=.55)
plt.title("Income vs Total Spending")
plt.xlabel("Income")
plt.ylabel("Total Spending")
plt.show()

# Correlation
corr_cols = [
    "Income","Recency","Total_Spending","Total_Purchases",
    "NumWebVisitsMonth","Digital_Engagement_Score","Estimated_CLV",
    "Churn_Risk_Score","Loyalty_Score","Customer_Profitability",
    "Satisfaction_Score","Response"
]
print(df[corr_cols].corr())

# ============================================================
# 4. LINEAR REGRESSION
# Business question:
# Which customer characteristics help explain total spending?
# ============================================================
lin_features = [
    "Income","Customer_Age","Total_Children","Recency",
    "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
    "NumStorePurchases","NumWebVisitsMonth","Campaign_Acceptance_Count",
    "Customer_Tenure_Months","Digital_Engagement_Score","Loyalty_Score"
]

X = model_df[lin_features]
y = model_df["Total_Spending"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)

pred = linear.predict(X_test)

r2 = r2_score(y_test, pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-len(lin_features)-1)
rmse = mean_squared_error(y_test, pred) ** .5
mae = mean_absolute_error(y_test, pred)

print("Linear Regression R2:", r2)
print("Adjusted R2:", adjusted_r2)
print("RMSE:", rmse)
print("MAE:", mae)

coefficients = pd.DataFrame({
    "Variable": lin_features,
    "Coefficient": linear.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

print(coefficients)

# Ridge and Lasso robustness checks
ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=10))
])
ridge.fit(X_train, y_train)
print("Ridge R2:", r2_score(y_test, ridge.predict(X_test)))

lasso = Pipeline([
    ("scale", StandardScaler()),
    ("model", Lasso(alpha=1.0, max_iter=10000))
])
lasso.fit(X_train, y_train)
print("Lasso R2:", r2_score(y_test, lasso.predict(X_test)))

# ============================================================
# 5. LOGISTIC REGRESSION
# Business question:
# Which customer characteristics are associated with campaign response?
# ============================================================
log_features = [
    "Income","Customer_Age","Total_Children","Recency",
    "NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",
    "NumStorePurchases","NumWebVisitsMonth","AcceptedCmp1",
    "AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5",
    "Complain","Campaign_Acceptance_Count","Customer_Tenure_Months",
    "Digital_Engagement_Score","Loyalty_Score","Satisfaction_Score"
]

X = model_df[log_features]
y = model_df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y
)

logistic = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

logistic.fit(X_train, y_train)

pred = logistic.predict(X_test)
prob = logistic.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall:", recall_score(y_test,pred))
print("ROC-AUC:", roc_auc_score(y_test,prob))
print("Confusion Matrix:")
print(confusion_matrix(y_test,pred))

# ============================================================
# 6. CLUSTER ANALYSIS
# Business question:
# Can customers be grouped into meaningful behavioral segments?
# ============================================================
cluster_features = [
    "Recency","Total_Spending","Total_Purchases",
    "NumWebPurchases","NumCatalogPurchases","NumStorePurchases",
    "Income","Digital_Engagement_Score","Loyalty_Score"
]

scaler = StandardScaler()
Z = scaler.fit_transform(model_df[cluster_features])

silhouette_scores = {}

for k in range(2,7):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Z)
    silhouette_scores[k] = silhouette_score(Z,labels)
    print("K =", k, "Silhouette =", silhouette_scores[k])

best_k = max(silhouette_scores, key=silhouette_scores.get)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)
df["Cluster"] = kmeans.fit_predict(Z)

print(df.groupby("Cluster")[cluster_features].mean())
print(df["Cluster"].value_counts())

# ============================================================
# 7. DECISION TREE
# Business question:
# What simple decision rules can identify likely campaign responders?
# ============================================================
tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
    class_weight="balanced"
)

tree.fit(X_train,y_train)

tree_pred = tree.predict(X_test)
tree_prob = tree.predict_proba(X_test)[:,1]

print("Decision Tree Accuracy:", accuracy_score(y_test,tree_pred))
print("Decision Tree Precision:", precision_score(y_test,tree_pred))
print("Decision Tree Recall:", recall_score(y_test,tree_pred))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test,tree_prob))

tree_importance = pd.DataFrame({
    "Variable": log_features,
    "Importance": tree.feature_importances_
}).sort_values("Importance",ascending=False)

print(tree_importance)

# ============================================================
# 8. RANDOM FOREST
# Business question:
# Which variables are most useful for predicting response?
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print("Random Forest Accuracy:", accuracy_score(y_test,rf_pred))
print("Random Forest Precision:", precision_score(y_test,rf_pred))
print("Random Forest Recall:", recall_score(y_test,rf_pred))
print("Random Forest ROC-AUC:", roc_auc_score(y_test,rf_prob))

rf_importance = pd.DataFrame({
    "Variable": log_features,
    "Importance": rf.feature_importances_
}).sort_values("Importance",ascending=False)

print(rf_importance)

# ============================================================
# 9. FINAL BUSINESS INTERPRETATION
# Use model results + EDA together.
# Do not interpret model accuracy alone; connect results to
# customer targeting, retention, engagement and channel strategy.
# ============================================================
