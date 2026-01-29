"""
Simple Logistic Classifier - Numpy Only Implementation.
Used as a lightweight alternative when XGBoost/TensorFlow not available.
"""
import numpy as np
from typing import Dict, List


class SimpleLogisticClassifier:
    """
    Simple multi-class logistic regression classifier.
    Implemented from scratch using only numpy.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, n_classes=3):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        self.classes_ = None
        
    def _softmax(self, z):
        """Softmax activation for multi-class."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot(self, y):
        """Convert labels to one-hot encoding."""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        for i, label in enumerate(y):
            one_hot[i, label] = 1
        return one_hot
    
    def fit(self, X, y):
        """Train the model."""
        n_samples, n_features = X.shape
        
        # Encode labels
        self.classes_ = np.unique(y)
        label_map = {label: i for i, label in enumerate(self.classes_)}
        y_encoded = np.array([label_map[label] for label in y])
        y_one_hot = self._one_hot(y_encoded)
        
        # Initialize weights
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            probs = self._softmax(z)
            
            # Compute gradients
            error = probs - y_one_hot
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0)
            
            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        z = np.dot(X, self.weights) + self.bias
        return self._softmax(z)
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
