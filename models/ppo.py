import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from utils.save_load import save_checkpoint, load_checkpoint

class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear_memory()

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n_states = len(self.states)
        if n_states == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []

        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in range(0, n_states, self.batch_size)]

        states = np.stack(self.states)
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        return states, actions, probs, vals, rewards, dones, batches

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ConvBackbone(nn.Module):
    def __init__(self, row_count, column_count, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        with T.no_grad():
            x = T.zeros(1, input_channels, row_count, column_count)
            x = self.conv1(x)
            x = self.conv2(x)
            flatten_size = x.view(1, -1).size(1)
        
        self.flatten_size = flatten_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return x

class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, lr, hidden_dim=128):
        super().__init__()
        row_count, col_count, input_channels = input_dims
        self.backbone = ConvBackbone(row_count, col_count, input_channels)
        self.fc1 = nn.Linear(self.backbone.flatten_size, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.backbone(state)
        x = F.relu(self.fc1(x))
        logits = self.fc_pi(x)
        dist = Categorical(logits=logits)
        return dist

    def save_model(self):
        save_checkpoint(self, self.optimizer, 'PPO/Actor.pth')

    def load_model(self):
        load_checkpoint(self, self.optimizer, 'PPO/Actor.pth', device=self.device)

class Critic(nn.Module):
    def __init__(self, input_dims, lr, hidden_dim=128):
        super().__init__()
        row_count, col_count, input_channels = input_dims
        self.backbone = ConvBackbone(row_count, col_count, input_channels)
        self.fc1 = nn.Linear(self.backbone.flatten_size, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.backbone(state)
        x = F.relu(self.fc1(x))
        value = self.fc_v(x)
        return value

    def save_model(self):
        save_checkpoint(self, self.optimizer, 'PPO/Critic.pth')

    def load_model(self):
        load_checkpoint(self, self.optimizer, 'PPO/Critic.pth', device=self.device)

class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.actor = Actor(n_actions, input_dims, lr)
        self.critic = Critic(input_dims, lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = T.tensor(np.array(observation, dtype=np.float32), device=self.actor.device)
        state = state.permute(2, 0, 1).unsqueeze(0)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        return (T.squeeze(action).item(),
                T.squeeze(dist.log_prob(action)).item(),
                T.squeeze(value).item())

    def learn(self):
        if len(self.memory.states) < self.memory.batch_size:
            return

        states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        if len(batches) == 0:
            return

        with T.no_grad():
            all_states = T.tensor(states, dtype=T.float, device=self.actor.device)
            if all_states.ndim == 4 and all_states.shape[-1] == self.actor.backbone.conv1.in_channels:
                all_states = all_states.permute(0, 3, 1, 2)
            values_tensor = self.critic(all_states).squeeze(-1)

        T_rewards = T.tensor(rewards, dtype=T.float, device=self.actor.device)
        T_dones = T.tensor(dones, dtype=T.float, device=self.actor.device)
        values = values_tensor.cpu().numpy()

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t+1] if t < len(rewards)-1 else 0.0
            delta = rewards[t] + self.gamma * next_value * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            advantages[t] = gae

        advantages = T.tensor(advantages, dtype=T.float, device=self.actor.device)

        if advantages.numel() > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            if not T.isfinite(adv_std) or adv_std.item() < 1e-8:
                adv_std = T.tensor(1.0, device=self.actor.device)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = T.zeros_like(advantages)

        advantages = T.clamp(advantages, -10.0, 10.0)

        old_probs_tensor = T.tensor(old_probs, dtype=T.float, device=self.actor.device)
        actions_arr = T.tensor(actions, dtype=T.long, device=self.actor.device)

        for _ in range(self.n_epochs):
            for batch_idx in batches:
                batch_states = T.tensor(states[batch_idx], dtype=T.float, device=self.actor.device)
                if batch_states.ndim == 4 and batch_states.shape[-1] == self.actor.backbone.conv1.in_channels:
                    batch_states = batch_states.permute(0, 3, 1, 2)

                batch_actions = actions_arr[batch_idx]
                batch_old_probs = old_probs_tensor[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_values = values_tensor[batch_idx].detach()

                dist = self.actor(batch_states)
                critic_val = self.critic(batch_states).squeeze(-1)

                if not T.isfinite(batch_advantages).all():
                    continue

                logits = None
                try:
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                except Exception as e:
                    continue

                prob_ratio = (new_log_probs - batch_old_probs).exp()

                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                actor_loss = -T.min(weighted_probs, weighted_clipped).mean() - 0.01 * entropy

                returns = batch_advantages + batch_values
                critic_loss = ((returns - critic_val) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                has_nan = False
                for p in list(self.actor.parameters()) + list(self.critic.parameters()):
                    if p.grad is not None and not T.isfinite(p.grad).all():
                        has_nan = True
                        break
                if has_nan:
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    continue

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()

    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
