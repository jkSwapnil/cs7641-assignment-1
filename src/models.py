# This module defines the model class

class Model:
    """Implements model"""

    def __init__(self, skl_model):
        """Constructor
        Parameters:
            skl_model: Object of sklearn object
        """
        self.skl_model = skl_model

    def fit(self, X, y):
        """Train the model
        Parameters :
            X: Features (np.ndarray[n_samples, n_features])
            y: Labels (np.ndarray[n_samples, ])
        """
        self.skl_model.fit(X, y)

    def predict_proba(self, X):
        """Output prediction probabaility
        Parameters:
            X: Input features (np.ndarray[n_samples, n_features])
        """
        return self.skl_model.predict_proba(X)

    def predict(self, X):
        """Output prediction
        Parameters:
            X: Input features (np.ndarray[n_samples, n_features])
        """
        return self.skl_model.predict(X)
