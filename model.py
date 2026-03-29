import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/resume.csv")

print("Before encoding:\n", df.dtypes)   # DEBUG

# Target
y = df['hired']

# Features
X = df.drop(['hired', 'candidate_id'], axis=1)

# Encode ALL categorical columns
encoders = {}

for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

print("\nAfter encoding:\n", X.dtypes)   # DEBUG

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("\n✅ Model trained successfully!")