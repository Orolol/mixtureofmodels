import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

class XGBInstructionClassifier:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.label_encoder = LabelEncoder()

    def train(self, features, labels, test_size=0.2, random_state=42):
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=random_state
        )

        # Define parameter grid for GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Perform GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Get the best model
        self.model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    def predict(self, features):
        return self.model.predict_proba(features)

    def predict_class(self, features):
        probabilities = self.predict(features)
        predicted_class = self.label_encoder.inverse_transform(probabilities.argmax(axis=1))
        return predicted_class[0]
