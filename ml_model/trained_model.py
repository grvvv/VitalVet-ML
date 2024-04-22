import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cattle health data
df = pd.read_csv('./cattlehealth.csv')

# Extract the features and labels
X = df[['temp', 'pulse', 'oxygen_saturation']]
y = df['health_status']

# Suppress the warning
warnings.filterwarnings("ignore", message="X does not have valid feature names*")

# Train the logistic regression model
model = RandomForestClassifier()
model.fit(X.values, y.values)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')
