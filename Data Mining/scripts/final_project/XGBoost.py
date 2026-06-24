import warnings
warnings.simplefilter('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os#Walking through directores
import plotly.graph_objects as go # Generate Graphs
from plotly.subplots import make_subplots #To Create Subplots
from sklearn import decomposition #pca
from sklearn.preprocessing import StandardScaler # Standardization ((X - X_mean)/X_std)
from sklearn.neighbors import KNeighborsClassifier #KNN Model
from sklearn.ensemble import RandomForestClassifier #RandomForest Model
from sklearn.linear_model import LogisticRegression #Logistic Model
from sklearn.model_selection import train_test_split # Splitting into train and test
from sklearn.model_selection import GridSearchCV# Hyperparameter Tuning
from sklearn.model_selection import cross_val_score#cross validation score
from sklearn.metrics import classification_report # text report showing the main classification metrics
from sklearn.metrics import confusion_matrix #to get confusion_matirx
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

missing_values = ['?', '--', ' ', 'NA', 'N/A', '-']
df = pd.read_csv('/Users/kao900531/Desktop/DataMining/cleaned_data_yearsBMI_2.csv', delimiter = ',', na_values = missing_values)
plt.rcParams['figure.figsize'] = (20, 15)
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="BuPu")
plt.title('Corelation Between Features', fontsize = 30)
plt.show()
#plt.savefig('/Users/kao900531/Desktop/DataMining/heatmap.png')

X = df.drop(['cardio', 'bmi', 'weight', 'gluc', 'gender', 'smoke', 'alco', 'active', 'height'], axis =1)
Y = df['cardio']

path = '/Users/kao900531/Desktop/DataMining/Cardio_processed_3.csv'
with open(path, 'w', encoding = 'utf-8-sig') as f:
  X.to_csv(f)

# Data Scaling
scaler = StandardScaler()
standard_X = scaler.fit_transform(X)

'''
# PCA
pca = decomposition.PCA(n_components=2)
pca_X = pca.fit_transform(standard_X)
pca_X = pd.DataFrame(pca_X)
pca_X.columns = ['P1', 'P2']
pca_X['cardio'] = Y
pca_X.head()
'''

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(standard_X, Y, test_size=0.25, random_state=42, shuffle = True)

# XGBoost
xgb_params = {
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5],
}

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_grid_cv = GridSearchCV(xgb_classifier, param_grid=xgb_params, cv=10)
xgb_grid_cv.fit(X_train, y_train)
print("Best Hyper Parameters:\n",xgb_grid_cv.best_params_)

# Evaluation
# Step 1: Evaluate the model using cross-validation
best_xgb = xgb_grid_cv.best_estimator_
scores = cross_val_score(best_xgb, X_train, y_train, cv=10)

# Step 2: Print the average accuracy and other statistics
print('Random Forest Model gives an average accuracy of {0:.2f} % with minimun of {1:.2f} % and maximum of {2:.2f} % accuracy'.format(scores.mean() * 100, scores.min() * 100, scores.max() * 100))

# Step 3: Make predictions on the test set
y_pred_xgb = best_xgb.predict(X_test)

# Step 4: Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot = True, linewidths=.5, cmap="viridis")
plt.title('Corelation Between Features')
plt.show()
plt.savefig('/Users/kao900531/Desktop/DataMining/XGBoost_heatmap.png')

#confusion_matrix = confusion_matrix(y_test, y_pred_xgb)
print('True Positive Cases : {}'.format(confusion_matrix(y_test, y_pred_xgb)[1][1]))
print('True Negative Cases : {}'.format(confusion_matrix(y_test, y_pred_xgb)[0][0]))
print('False Positive Cases : {}'.format(confusion_matrix(y_test, y_pred_xgb)[0][1]))
print('False Negative Cases : {}'.format(confusion_matrix(y_test, y_pred_xgb)[1][0]))

conf_matrix = confusion_matrix(y_test, y_pred_xgb)
true_positive = conf_matrix[1, 1]
false_positive = conf_matrix[0, 1]
true_negative = conf_matrix[0, 0]
false_negative = conf_matrix[1, 0]

sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
f1_score = 2 * (true_positive / (2 * true_positive + false_positive + false_negative))

print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
print(f"Specificity (True Negative Rate): {specificity:.2f}")
print(f"F1 Score: {f1_score:.2f}")

