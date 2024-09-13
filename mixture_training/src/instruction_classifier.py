import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5):
        super(ImprovedNN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        for i, (layer, bn, dropout) in enumerate(zip(self.layers[:-1], self.batch_norms, self.dropouts)):
            x = layer(x)
            x = bn(x)
            x = nn.ReLU()(x)
            x = dropout(x)
        x = self.layers[-1](x)
        return x

class InstructionClassifier:
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_classes=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = ImprovedNN(input_size, hidden_sizes, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.label_encoder = LabelEncoder()
        
    def train(self, features, labels, num_epochs=200, batch_size=64, validation_split=0.2, patience=20):
        # Split the data into training and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            features, labels, test_size=validation_split, random_state=42
        )
        
        train_features = torch.FloatTensor(train_features).to(self.device)
        val_features = torch.FloatTensor(val_features).to(self.device)
        
        self.label_encoder.fit(labels)
        encoded_train_labels = self.label_encoder.transform(train_labels)
        encoded_val_labels = self.label_encoder.transform(val_labels)
        
        train_labels = torch.LongTensor(encoded_train_labels).to(self.device)
        val_labels = torch.LongTensor(encoded_val_labels).to(self.device)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(train_features), batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / (len(train_features) // batch_size)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_features)
                val_loss = self.criterion(val_outputs, val_labels)
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    # Load the best model
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
    
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    def predict_class(self, features):
        probabilities = self.predict(features)
        predicted_class = self.label_encoder.inverse_transform(probabilities.argmax(axis=1))
        return predicted_class[0]
