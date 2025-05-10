import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import DataPreprocessor
from sentiment_model import SentimentClassifier, SentimentTrainer
from rl_trainer import RLEnvironment, PPOAgent, RLTrainer
import wandb

def create_data_loaders(preprocessor, batch_size=32):
    # Prepara i dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.prepare_dataset()
    
    # Crea i data loaders
    train_dataset = TensorDataset( 
        torch.cat([x['input_ids'] for x in X_train]),
        torch.cat([x['attention_mask'] for x in X_train]),
        torch.tensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.cat([x['input_ids'] for x in X_val]),
        torch.cat([x['attention_mask'] for x in X_val]),
        torch.tensor(y_val)
    )
    
    test_dataset = TensorDataset(
        torch.cat([x['input_ids'] for x in X_test]),
        torch.cat([x['attention_mask'] for x in X_test]),
        torch.tensor(y_test)
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def main():
    # Inizializza wandb per il tracking degli esperimenti
    wandb.init(project="amazon-sentiment-rl", name="sentiment-rl-experiment")
    
    # Imposta il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepara i dati
    preprocessor = DataPreprocessor(category="Toys_and_Games")
    train_loader, val_loader, test_loader = create_data_loaders(preprocessor)
    
    # Inizializza il modello di sentiment analysis
    sentiment_model = SentimentClassifier().to(device)
    sentiment_trainer = SentimentTrainer(sentiment_model, device)
    
    # Training supervisionato iniziale
    print("Iniziando il training supervisionato...")
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = sentiment_trainer.train_epoch(train_loader)
        val_loss, val_acc = sentiment_trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    
    # Inizializza l'ambiente RL e l'agente
    rl_env = RLEnvironment(sentiment_model, val_loader, device)
    state_dim = sentiment_model.fc.in_features
    action_dim = sentiment_model.fc.out_features
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    rl_trainer = RLTrainer(rl_env, ppo_agent, device)
    
    # Training con reinforcement learning
    print("\nIniziando il fine-tuning con RL...")
    num_episodes = 100
    for episode in range(num_episodes):
        total_reward = rl_trainer.train_episode()
        
        # Valuta le performance
        val_loss, val_acc = sentiment_trainer.evaluate(val_loader)
        
        print(f"Episode {episode+1}/{num_episodes}:")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "episode": episode,
            "total_reward": total_reward,
            "rl_val_acc": val_acc
        })
    
    # Valutazione finale sul test set
    test_loss, test_acc = sentiment_trainer.evaluate(test_loader)
    print(f"\nPerformance finale sul test set:")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    wandb.log({"final_test_acc": test_acc})
    wandb.finish()

if __name__ == "__main__":
    main()