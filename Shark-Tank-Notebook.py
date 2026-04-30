# IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
data = pd.read_csv('shark_tank_companies.csv')

print("Initial shape:", data.shape)
print("Missing values:\n", data.isnull().sum())

# fill missing non-critical values
data['website'] = data['website'].fillna('Unknown')
data['entrepreneurs'] = data['entrepreneurs'].fillna('Unknown')

# convert numeric columns
numeric_cols = ['askedfor', 'exchangeforstake', 'valuation']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# drop rows with missing critical values
data = data.dropna(subset=numeric_cols)

# convert target variable
data['deal'] = data['deal'].astype(int)

# remove irrelevant columns
data = data.drop(columns=[
    'description',
    'website',
    'title',
    'episode_season',
    'entrepreneurs',
    'location'
])

# create number of sharks feature
data['num_sharks'] = data[['shark1','shark2','shark3','shark4','shark5']].notnull().sum(axis=1)

# drop individual shark columns
data = data.drop(columns=['shark1','shark2','shark3','shark4','shark5'])

# apply log transformation
data['askedfor'] = np.log1p(data['askedfor'])
data['valuation'] = np.log1p(data['valuation'])

# discretize asked amount
data['ask_level'] = pd.cut(data['askedfor'], bins=3, tick_labels=['Low', 'Medium', 'High'])

# encode discretized feature
data = pd.get_dummies(data, columns=['ask_level'], drop_first=True)

# encode category
data = pd.get_dummies(data, columns=['category'], drop_first=True)

# final checks
print("\nFinal shape:", data.shape)
print("Remaining missing values:\n", data.isnull().sum())
print("Class balance:\n", data['deal'].value_counts(normalize=True))
print("Final columns:\n", data.columns)

# correlation heatmap
cols = ['deal', 'askedfor', 'exchangeforstake', 'valuation', 'num_sharks']
plt.figure(figsize=(10, 7))
sns.heatmap(data[cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# boxplot for asked amount vs deal
plt.figure(figsize=(8, 6))
plt.boxplot([
    data[data['deal'] == 0]['askedfor'],
    data[data['deal'] == 1]['askedfor']
], tick_labels=['No Deal', 'Deal'])
plt.title('Asked Amount vs Deal Outcome')
plt.ylabel('Log(Asked Amount)')
plt.xlabel('Deal Outcome')
plt.show()

# boxplot for equity vs deal
plt.figure(figsize=(8, 6))
plt.boxplot([
    data[data['deal'] == 0]['exchangeforstake'],
    data[data['deal'] == 1]['exchangeforstake']
], tick_labels=['No Deal', 'Deal'])
plt.title('Equity Offered vs Deal Outcome')
plt.ylabel('Equity Offered (%)')
plt.xlabel('Deal Outcome')
plt.show()

# reload raw data for category analysis
raw = pd.read_csv('shark_tank_companies.csv')
raw['deal'] = raw['deal'].astype(int)

# filter categories with enough samples
counts = raw['category'].value_counts()
valid_categories = counts[counts >= 5].index
filtered = raw[raw['category'].isin(valid_categories)]

# compute deal rate by category
deal_rate = filtered.groupby('category')['deal'].mean().sort_values(ascending=False)

# plot category deal rates
plt.figure(figsize=(12,6))
deal_rate.plot(kind='bar')
plt.title('Deal Rate by Category')
plt.ylabel('Deal Rate')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.show()

# MODELS
# Deal Classification Models
# define features and target
X = data.drop(columns=['deal', 'valuation'])
y = data['deal']

# set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# logistic regression model
log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# random forest model
rf_model = RandomForestClassifier(random_state=42)

# cross-validation accuracy scores
log_scores = cross_val_score(log_model, X, y, cv=cv, scoring='accuracy')
rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')

print("Logistic Regression CV Accuracy Scores:", log_scores)
print("Logistic Regression Mean Accuracy:", log_scores.mean())

print("Random Forest CV Accuracy Scores:", rf_scores)
print("Random Forest Mean Accuracy:", rf_scores.mean())

# cross-validated predictions for deeper evaluation
y_pred_log = cross_val_predict(log_model, X, y, cv=cv)
y_pred_rf = cross_val_predict(rf_model, X, y, cv=cv)

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y, y_pred_log))

print("\nLogistic Regression Classification Report:")
print(classification_report(y, y_pred_log))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y, y_pred_rf))

# REGRESSION MODELS
# define features and target
X_reg = data.drop(columns=['valuation', 'deal'])
y_reg = data['valuation']

# cross-validation setup
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

# linear regression (with scaling)
lin_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# random forest regressor
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

# cross-validation scores (R²)
lin_scores = cross_val_score(lin_model, X_reg, y_reg, cv=cv_reg, scoring='r2')
rf_scores = cross_val_score(rf_reg, X_reg, y_reg, cv=cv_reg, scoring='r2')

print("\nLinear Regression R² Scores:", lin_scores)
print("Linear Regression Mean R²:", lin_scores.mean())

print("\nRandom Forest R² Scores:", rf_scores)
print("Random Forest Mean R²:", rf_scores.mean())

# predictions for evaluation
y_pred_lin = cross_val_predict(lin_model, X_reg, y_reg, cv=cv_reg)
y_pred_rf = cross_val_predict(rf_reg, X_reg, y_reg, cv=cv_reg)

# evaluation metrics
print("\nLinear Regression MSE:", mean_squared_error(y_reg, y_pred_lin))
print("Linear Regression R²:", r2_score(y_reg, y_pred_lin))

print("\nRandom Forest MSE:", mean_squared_error(y_reg, y_pred_rf))
print("Random Forest R²:", r2_score(y_reg, y_pred_rf))

# feature importance from random forest regressor
rf_reg.fit(X_reg, y_reg)
importances = pd.Series(rf_reg.feature_importances_, index=X_reg.columns)

print("\nTop Features for Valuation Prediction:")
print(importances.sort_values(ascending=False).head(10))

# New dataset for classification comparison
data2 = pd.read_csv('Shark Tank India Dataset.csv')

data2['deal'] = data2['deal'].astype(int)

data2 = data2.drop(columns=[
    'deal_amount',
    'deal_equity',
    'deal_valuation',
    'ashneer_deal',
    'anupam_deal',
    'aman_deal',
    'namita_deal',
    'vineeta_deal',
    'peyush_deal',
    'ghazal_deal',
    'total_sharks_invested',
    'amount_per_shark',
    'equity_per_shark'
])

tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(data2['idea'])

text_df = pd.DataFrame(text_features.toarray(), columns=tfidf.get_feature_names_out())

data2 = pd.concat([data2.reset_index(drop=True), text_df], axis=1)
data2 = data2.drop(columns=['idea'])

X2 = data2.drop(columns=['deal'])
y2 = data2['deal']

# drop remaining text columns like brand_name
X2 = X2.select_dtypes(exclude=['object'])

# fill missing numeric values
X2 = X2.fillna(0)

# set up cross-validation
cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# logistic regression model for new dataset
log_model2 = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# random forest model for new dataset
rf_model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    random_state=42
)

# cross-validation accuracy scores
log_scores2 = cross_val_score(log_model2, X2, y2, cv=cv2, scoring='accuracy')
rf_scores2 = cross_val_score(rf_model2, X2, y2, cv=cv2, scoring='accuracy')

print("\nShark Tank India Logistic Regression CV Accuracy Scores:", log_scores2)
print("Shark Tank India Logistic Regression Mean Accuracy:", log_scores2.mean())

print("\nShark Tank India Random Forest CV Accuracy Scores:", rf_scores2)
print("Shark Tank India Random Forest Mean Accuracy:", rf_scores2.mean())

# cross-validated predictions
y_pred_log2 = cross_val_predict(log_model2, X2, y2, cv=cv2)
y_pred_rf2 = cross_val_predict(rf_model2, X2, y2, cv=cv2)

print("\nShark Tank India Logistic Regression Confusion Matrix:")
print(confusion_matrix(y2, y_pred_log2))

print("\nShark Tank India Logistic Regression Classification Report:")
print(classification_report(y2, y_pred_log2))

print("\nShark Tank India Random Forest Confusion Matrix:")
print(confusion_matrix(y2, y_pred_rf2))

print("\nShark Tank India Random Forest Classification Report:")
print(classification_report(y2, y_pred_rf2))