

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import joblib  # Use joblib instead of pickle
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

# Prepare the features and labels
X = data[1:, 1:-1]
print(X)
y = data[1:, -1]
print(y)
y = y.astype('int')
X = X.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Example input for testing
inputt = [int(x) for x in "45 32 60".split(' ')]
final = [np.array(inputt)]

# Test the prediction
b = log_reg.predict_proba(final)

# Save the model using joblib
joblib.dump(log_reg, 'model.pkl')

# Load the model using joblib to verify (optional)
model = joblib.load('model.pkl')
