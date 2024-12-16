import pickle
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Fetch the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Reduce the dataset size for memory efficiency
X, y = X[:1000], y[:1000]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate the classifier
print(clf.score(X_test, y_test))

# Save the trained model to a file
with open('mnist_model.pk', 'wb') as f:
    pickle.dump(clf, f)  # Fixed typo here
