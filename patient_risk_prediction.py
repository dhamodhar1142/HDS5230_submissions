import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('diabetes.csv')

# Data cleaning
if df.isnull().any().any():
    df = df.fillna(df.median())

# Feature selection
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Model training with hyperparameter tuning
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid={'n_estimators':[50,100,150], 'max_depth':[None,5,10]}, cv=3)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Prediction and evaluation
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:,1]
accuracy = accuracy_score(y_test, preds)
roc_score = roc_auc_score(y_test, preds_proba)
report = classification_report(y_test, preds)

# Visualization - feature importances
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
feat_imp.plot(kind='bar')
plt.title('Feature Importances - Diabetes Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Save model
joblib.dump(model, 'diabetes_patient_risk_model.pkl')

# Save results
with open('results.txt','w') as f:
    f.write('Classification Report:\n' + report + f'\nAccuracy: {accuracy:.2f}\nROC AUC: {roc_score:.2f}\n')

print('Model trained. Accuracy:', accuracy)
print('ROC AUC:', roc_score)
print('Feature importances plot saved.')
