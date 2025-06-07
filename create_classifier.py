# create_classifier.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate mock embeddings + labels
X = np.random.randn(200, 512)
y = np.random.choice(['British', 'American', 'Australian'], 200)

# Train simple classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Saved classifier.pkl")


