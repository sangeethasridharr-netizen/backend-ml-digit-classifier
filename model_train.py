# model_train.py
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("Loading MNIST dataset... (This might take a moment)")
    # Using version 1 of MNIST
    # As_frame=False to get numpy arrays directly
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    X, y = mnist.data, mnist.target
    
    # Scale pixels to [0, 1]
    X = X / 255.0
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize Random Forest Classifier
    # n_estimators=100 is a good default, but maybe less for speed in demo
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
