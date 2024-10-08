from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, TrainerCallback, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup, RobertaForSequenceClassification, BertForSequenceClassification, Trainer, TrainingArguments
import warnings
from torch.optim import AdamW
import logging
from collections import Counter

from nltk.corpus import stopwords
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if state.global_step % 100 == 0:
                logger.info(f"Step {state.global_step}: {logs}")

class InstructionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def preprocess_text(self, text):
        # convert to string
        text = str(text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        # remove stop words
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
            text = self.preprocess_text(text)
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
    def __init__(self, num_classes, model_type='roberta-large', class_weights=None):
        super(TransformerClassifier, self).__init__()
        if 'roberta' in model_type:
            self.model = RobertaForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)
        elif 'bert' in model_type:
            self.model = BertForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_weights = class_weights.to(self.device)
        
    def forward(self, input_ids, attention_mask, labels=None):
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return loss_fct(outputs.logits, labels), outputs

class InstructionClassifier:
    def __init__(self, num_classes=20, max_length=128, model_type='roberta-large', best_model_path=None, class_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        if 'roberta' in model_type:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_type, use_fast=True)
        elif 'bert' in model_type:
            self.tokenizer = BertTokenizer.from_pretrained(model_type, use_fast=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = TransformerClassifier(num_classes, model_type, class_weights).to(self.device)
        if best_model_path:
            self.load_model(best_model_path)
        
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
        
    def train(self, texts, labels, num_epochs=5, batch_size=8, learning_rate=1e-5, validation_split=0.05):
        # Calculate and display original class distribution
        self.calculate_class_distribution(labels)
        logger.info(f"Starting training with {len(texts)} samples")
        
        # Update your label encoder
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=validation_split, random_state=42, stratify=encoded_labels
        )
        
        logger.info(f"Train set size: {len(train_texts)}")
        logger.info("Train set class distribution:")
        self.calculate_class_distribution(self.label_encoder.inverse_transform(train_labels))
        
        logger.info(f"Validation set size: {len(val_texts)}")
        logger.info("Validation set class distribution:")
        self.calculate_class_distribution(self.label_encoder.inverse_transform(val_labels))
        
        # Create datasets
        train_dataset = InstructionDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = InstructionDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps) 
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            total_train_loss_not_weighted = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                
                    self.model.zero_grad()
                    results = self.model(input_ids, attention_mask, labels)
                    loss = results[0]
                    out = results[1]
                    outputs = out.logits
                    loss_not_weighted = out.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    scheduler.step()
                
                    total_train_loss += loss.item()
                    total_train_loss_not_weighted += loss_not_weighted.item()
                    if batch_idx % 50 == 0 and batch_idx != 0:
                        # Calculate accuracy and F1 score
                        _, preds = torch.max(F.softmax(outputs, dim=1), dim=1)
                        
                        accuracy = (preds == labels).float().mean()
                        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Loss Not Weighted: {loss_not_weighted.item():.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

                    # validte every 20% of the training data
                    if batch_idx % (len(train_loader) * 0.2) == 0 and batch_idx != 0:
                        avg_train_loss = total_train_loss / len(train_loader)
                        val_loss, f1, accuracy = self.validate(val_texts, val_labels)
                        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}')
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(self.model.state_dict(), 'best_roberta_model.pth')
                            logger.info("Saved best model.")

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

        # Load the best model
        self.model.load_state_dict(torch.load('best_roberta_model.pth'))
        logger.info("Training completed.")
        
    def validate(self, texts, labels):
        logger.info(f"Validating {len(texts)} samples")
        self.model.eval()
        dataset = InstructionDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=1)
        predictions = []
        total_loss = 0
        f1 = 0
        accuracy = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                results = self.model(input_ids, attention_mask, labels)
                loss = results[0]
                out = results[1]
                outputs = out.logits
                total_loss += loss.item()
                
                _, preds = torch.max(F.softmax(outputs, dim=1), dim=1)
                f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                accuracy += (preds == labels).float().mean()

        return total_loss / len(dataloader), f1 / len(dataloader), accuracy / len(dataloader)
    
    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        f1 = f1_score(labels, preds, average='weighted')
        accuracy = (preds == labels).mean()
        logger.info(f"Accuracy: {accuracy}, F1: {f1}")
        return {'accuracy': accuracy, 'f1': f1}
    
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
                _, preds = torch.max(F.softmax(outputs, dim=1), dim=1)
                predictions.extend(preds.cpu().tolist())
        logger.info(f"Predictions: {predictions}")
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_class(self, text):
        return self.predict([text])[0]

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Loaded best model from {model_path}")
