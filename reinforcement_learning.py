import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import heapq

# Neural Network for Deep Q-Learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64], dropout_rate=0.1):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Build a deeper network with configurable hidden layers
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))     # Add dropout for regularization
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Prioritized Experience Replay Memory (simplified)
class PriorityReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.memory = []    # Heap for priorities
        self.experiences = deque(maxlen=capacity)  # Store experiences
        self.max_priority = 1.0

    def add(self, experience, error=None):
        priority = error if error is not None else self.max_priority
        priority = (abs(priority) + 1e-5) ** self.alpha  # Small constant to avoid zero priority
        heapq.heappush(self.memory, (-priority, len(self.experiences)))  # Negative for max heap
        self.experiences.append(experience)

    def sample(self, batch_size):
        if len(self.experiences) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = np.array([-p for p, _ in self.memory[:len(self.experiences)]])
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.experiences), batch_size, p=probs, replace=False)
        samples = [self.experiences[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.experiences) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.memory[idx] = (-priority, self.memory[idx][1])
            self.max_priority = max(self.max_priority, priority)
        heapq.heapify(self.memory)  # Re-heapify after updates

    def __len__(self):
        return len(self.experiences)

# Enhanced Reinforcement Learning Agent
class RLAgent:
    def __init__(self, state_size, action_size, 
                 lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 memory_capacity=10000, batch_size=64, target_update_freq=1000,
                 use_double_dqn=True, clip_grad_norm=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.clip_grad_norm = clip_grad_norm
        
        # Networks
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Target network doesn't train
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Memory
        self.memory = PriorityReplayMemory(memory_capacity)
        self.steps = 0
        self.target_update_freq = target_update_freq

    def remember(self, state, action, reward, next_state, done):
        # Initial error estimate (could be refined with TD error later)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            q_value = self.policy_net(state_tensor)[action]
            next_q = self.target_net(next_state_tensor).max().item()
            target = reward + (1 - done) * self.gamma * next_q
            error = abs(q_value.item() - target)
        self.memory.add((state, action, reward, next_state, done), error)

    def act(self, state):
        self.steps += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from memory
        batch, indices, weights = self.memory.sample(self.batch_size)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = weights.unsqueeze(1)
        
        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN or standard DQN
        if self.use_double_dqn:
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute TD errors for priority update
        td_errors = (q_values - targets).detach().cpu().numpy()
        
        # Loss with importance sampling weights
        loss = (self.criterion(q_values, targets) * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)