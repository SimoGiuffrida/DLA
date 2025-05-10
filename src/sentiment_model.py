import torch
import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class SentimentTrainer:
    def __init__(self, model, device, learning_rate=2e-5, patience=3):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model = None
        
        # Importa le configurazioni per il salvataggio
        from config import CHECKPOINT_DIR, SAVE_FREQUENCY, MAX_CHECKPOINTS
        self.checkpoint_dir = CHECKPOINT_DIR
        self.save_frequency = SAVE_FREQUENCY
        self.max_checkpoints = MAX_CHECKPOINTS
        self.checkpoint_files = []
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_files.append(checkpoint_path)
        
        # Mantieni solo gli ultimi N checkpoint
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc='Training', leave=True)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Aggiorna la barra di progresso con le metriche correnti
            avg_loss = total_loss / (progress_bar.n + 1)
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.4f}'})
            
        progress_bar.close()
        return total_loss / len(train_loader), correct_predictions / total_predictions
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        from tqdm import tqdm
        progress_bar = tqdm(val_loader, desc='Validazione', leave=True)
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Aggiorna la barra di progresso con le metriche correnti
                avg_loss = total_loss / (progress_bar.n + 1)
                accuracy = correct_predictions / total_predictions
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.4f}'})
            
        progress_bar.close()
        val_loss = total_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model = self.model.state_dict()
        else:
            self.counter += 1
        
        return val_loss, val_accuracy