import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class InstructionClassifier:
    def __init__(self, input_size, hidden_size=64, num_classes=10):
        self.model = SimpleNN(input_size, hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.label_encoder = LabelEncoder()
        
    def train(self, train_features, train_labels, num_epochs=10, batch_size=32):
        train_features = torch.FloatTensor(train_features)
        encoded_labels = self.label_encoder.fit_transform(train_labels)
        train_labels = torch.LongTensor(encoded_labels)
        
        for epoch in range(num_epochs):
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()
    
    def predict_class(self, features):
        probabilities = self.predict(features)
        predicted_class = self.label_encoder.inverse_transform(probabilities.argmax(axis=1))
        return predicted_class[0]
