import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class Conv_QNet(nn.Module):
    def __init__(self, row_count, column_count, input_channels, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        with torch.no_grad():
            x = torch.zeros(1, input_channels, row_count, column_count)
            x = self.conv1(x)
            x = self.conv2(x)
            flatten_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayMemory:
    def __init__(self, capacity, rng, device):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng
        self.device = device

    def push(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (torch.stack(states), 
                torch.stack(actions).long(), 
                torch.stack(rewards).float(), 
                torch.stack(next_states), 
                torch.stack(dones).bool())
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size, params, rng, device):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = params['gamma']
        self.rng = rng
        self.device = device
        input_channels = state_shape[2]
        row_count, column_count = state_shape[0], state_shape[1]
        self.policy_net = Conv_QNet(row_count, column_count, input_channels, action_size).to(device)
        self.target_net = Conv_QNet(row_count, column_count, input_channels, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params['learning_rate'])
        self.memory = ReplayMemory(params['max_memory'], self.rng, device)
        self.steps_done = 0

    def choose_action(self, state, exploration_rate):
        if self.rng.random() < exploration_rate:
            return self.rng.integers(0, self.action_size)  
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
            
    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
