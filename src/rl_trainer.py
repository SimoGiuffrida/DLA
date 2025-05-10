import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RLEnvironment:
    def __init__(self, sentiment_model, val_loader, device):
        self.sentiment_model = sentiment_model
        self.val_loader = val_loader
        self.device = device
        self.current_step = 0
        self.max_steps = len(val_loader)

    def reset(self):
        self.current_step = 0
        batch = next(iter(self.val_loader))
        return self._get_state(batch)

    def step(self, action):
        batch = next(iter(self.val_loader))
        original_pred = self._get_prediction(batch)
        
        # Applica l'azione (modifica dei pesi del modello)
        self._apply_action(action)
        
        new_pred = self._get_prediction(batch)
        reward = self._calculate_reward(original_pred, new_pred, batch['labels'])
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(batch), reward, done

    def _get_state(self, batch):
        with torch.no_grad():
            outputs = self.sentiment_model(batch['input_ids'].to(self.device),
                                         batch['attention_mask'].to(self.device))
            probs = F.softmax(outputs, dim=1)
            return probs

    def _get_prediction(self, batch):
        with torch.no_grad():
            outputs = self.sentiment_model(batch['input_ids'].to(self.device),
                                         batch['attention_mask'].to(self.device))
            return torch.argmax(outputs, dim=1)

    def _apply_action(self, action):
        # Implementa la logica per modificare i pesi del modello
        # basata sull'azione scelta dall'agente
        pass

    def _calculate_reward(self, original_pred, new_pred, true_labels):
        original_correct = (original_pred == true_labels.to(self.device)).float().mean()
        new_correct = (new_pred == true_labels.to(self.device)).float().mean()
        return new_correct - original_correct

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        action_probs = F.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return action_probs, value

class RLTrainer:
    def __init__(self, env, agent, device, learning_rate=3e-4, gamma=0.99, epsilon=0.2):
        self.env = env
        self.agent = agent
        self.device = device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def train_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Ottieni l'azione dall'agente
            action_probs, value = self.agent(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Fai uno step nell'ambiente
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            
            # Calcola il vantaggio e aggiorna i pesi
            next_value = self.agent.critic(next_state)
            advantage = reward + (self.gamma * next_value * (1 - done)) - value
            
            # Calcola le loss per actor e critic
            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            
            # Aggiorna i pesi
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            state = next_state
        
        return total_reward