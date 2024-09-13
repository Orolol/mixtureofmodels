import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

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
        
    def train(self, features, labels, num_epochs=100, batch_size=32, validation_split=0.2, patience=10):
        # Split the data into training and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            features, labels, test_size=validation_split, random_state=42
        )
        
        train_features = torch.FloatTensor(train_features)
        val_features = torch.FloatTensor(val_features)
        
        encoded_train_labels = self.label_encoder.fit_transform(train_labels)
        encoded_val_labels = self.label_encoder.transform(val_labels)
        
        train_labels = torch.LongTensor(encoded_train_labels)
        val_labels = torch.LongTensor(encoded_val_labels)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_features)
                val_loss = self.criterion(val_outputs, val_labels)
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
    
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
