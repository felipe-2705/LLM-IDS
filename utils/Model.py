from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
from utils.Logger import CustomLogger
from utils.Algorithm import ALGORITHM
from utils.DataBase import DataBase

class Model():
    def __init__(self, algorithm: str = 'nb', database: DataBase=None, logger: CustomLogger=None):
        self.logging = logger
        self.algorithm= algorithm
        self.db = database
        self.set_classifier(algorithm)

    def set_classifier(self, algorithm: str):
        algorithm_dict =  ALGORITHM().algorithms
        if algorithm not in algorithm_dict:
            valid_options = ", ".join(algorithm_dict.keys())
            raise ValueError(f"Unsupported algorithm. Valid options are: {valid_options}")
        self.classifier = algorithm_dict[algorithm]()

    def calculate_metrics(self,y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1

    def select_data(self, features):
        # Select features from the training and test datasets
        X_train = self.db.x_train[features]
        y_train = self.db.y_train
        X_test = self.db.x_test[features]
        y_test = self.db.y_test

        return X_train, y_train, X_test, y_test

    def evaluate_model(self,features):
        # Train the classifier
        X_train, y_train, X_test, y_test = self.select_data(features)
        self.classifier.fit(X_train, y_train)

        # Predict labels for the test dataset
        y_pred = self.classifier.predict(X_test)

        # Calculate metrics
        f1 = self.calculate_metrics(y_test, y_pred)

        return f1
    
    def evaluate_algorithm(self,features_idx):
        features = [self.db.feature_names[i] for i in features_idx]
        return self.evaluate_model(features)

    def evaluate_baseline(self):
        self.logging.info("\nBaseline Evaluation with all features using the selected algorithm:")
        f1 = self.evaluate_algorithm(list(range(len(self.db.feature_names))))
        self.logging.info(f"Baseline F1-Score ({self.algorithm.upper()}): {f1:.4f}")
        self.logging.info("-" * 50)
        return f1