from sklearn.ensemble import RandomForestClassifier

class ModelRecommender:
    def __init__(self):
        self.model = RandomForestClassifier()
        
    def train(self, features, labels):
        self.model.fit(features, labels)
    
    def predict(self, features):
        return self.model.predict_proba(features)
