from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup, RobertaPooler
import warnings
from torch.optim import AdamW
import logging
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

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

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, model_type='roberta-large'):
        super(TransformerClassifier, self).__init__()
        if 'roberta' in model_type:
            config = RobertaConfig.from_pretrained(model_type)
            self.transformer = RobertaModel.from_pretrained(model_type, config=config)
        elif 'bert' in model_type:
            config = BertConfig.from_pretrained(model_type)
            self.transformer = BertModel.from_pretrained(model_type, config=config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        
        if self.transformer.pooler is None and 'roberta' in model_type:
            self.transformer.pooler = RobertaPooler(self.transformer.config)
            self.transformer.pooler.apply(self.transformer._init_weights)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class InstructionClassifier:
    def __init__(self, num_classes=20, max_length=128, model_type='roberta-large'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        if 'roberta' in model_type:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_type, use_fast=True)
        elif 'bert' in model_type:
            self.tokenizer = BertTokenizer.from_pretrained(model_type, use_fast=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = TransformerClassifier(num_classes, model_type).to(self.device)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.model_type = model_type
        
    def calculate_class_distribution(self, labels):
        class_distribution = Counter(labels)
        total_samples = len(labels)
        
        logger.info("Class Distribution:")
        for class_label, count in class_distribution.items():
            percentage = (count / total_samples) * 100
            logger.info(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        return class_distribution
        
    def train(self, texts, labels, num_epochs=5, batch_size=8, learning_rate=1e-5, validation_split=0.2):
        # Calculate and display original class distribution
        self.calculate_class_distribution(labels)
        logger.info(f"Starting training with {len(texts)} samples")
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=validation_split, random_state=42, stratify=encoded_labels
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        logger.info(f"Class weights: {class_weights}")
        
        logger.info(f"Train set size: {len(train_texts)}")
        logger.info("Train set class distribution:")
        self.calculate_class_distribution(self.label_encoder.inverse_transform(train_labels))
        
        logger.info(f"Validation set size: {len(val_texts)}")
        logger.info("Validation set class distribution:")
        self.calculate_class_distribution(self.label_encoder.inverse_transform(val_labels))
        
        # Create datasets
        train_dataset = InstructionDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = InstructionDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps) 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
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
                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    scheduler.step()
                
                    total_train_loss += loss.item()

                    if batch_idx % 100 == 0 and batch_idx != 0:
                        # Calculate accuracy and F1 score
                        _, preds = torch.max(outputs, dim=1)
                        accuracy = (preds == labels).float().mean()
                        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                        print(labels.cpu().numpy(), preds.cpu().numpy())
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
                        loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
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
