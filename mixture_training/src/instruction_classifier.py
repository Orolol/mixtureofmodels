from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, get_linear_schedule_with_warmup
import warnings
from torch.optim import AdamW
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

class InstructionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
            label = self.labels[idx]
        except IndexError as e:
            logger.error(f"Index {idx} is out of bounds. Dataset length: {len(self.texts)}")
            raise IndexError(f"Index {idx} is out of bounds. Dataset length: {len(self.texts)}") from e
        except Exception as e:
            logger.error(f"Error accessing data at index {idx}: {str(e)}")
            raise

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaClassifier, self).__init__()
        config = RobertaConfig.from_pretrained('roberta-large')
        self.roberta = RobertaModel.from_pretrained('roberta-large', config=config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class InstructionClassifier:
    def __init__(self, num_classes=20, max_length=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_fast=True)
        self.model = RoBERTaClassifier(num_classes).to(self.device)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
    def train(self, texts, labels, num_epochs=5, batch_size=8, learning_rate=5e-5, validation_split=0.2):
        logger.info(f"Starting training with {len(texts)} samples")
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=validation_split, random_state=42
        )
        
        logger.info(f"Train set size: {len(train_texts)}, Validation set size: {len(val_texts)}")
        
        # Create datasets
        train_dataset = InstructionDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = InstructionDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                
                    self.model.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                
                    total_train_loss += loss.item()

                    if batch_idx % 100 == 0 and batch_idx != 0:
                        # Calculate accuracy and F1 score
                        _, preds = torch.max(outputs, dim=1)
                        accuracy = (preds == labels).float().mean()
                        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                        
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")



                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        outputs = self.model(input_ids, attention_mask)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        total_val_loss += loss.item()

                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_roberta_model.pth')
                logger.info("Saved best model.")
        
        # Load the best model
        self.model.load_state_dict(torch.load('best_roberta_model.pth'))
        logger.info("Training completed.")
    
    def predict(self, texts):
        self.model.eval()
        logger.info(f"Predicting {len(texts)} samples")
        dataset = InstructionDataset(texts, [0]*len(texts), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=1)
        logger.info(f"Dataloader: {dataloader}")
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
        logger.info(f"Predictions: {predictions}")
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_class(self, text):
        return self.predict([text])[0]
