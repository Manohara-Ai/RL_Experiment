import numpy as np
import torch as T
import torch.nn as nn
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
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in range(0, n_states, self.batch_size)]

        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches)

    def clear_memory(self):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = [], [], [], [], [], []

class Actor(nn.Module):
    def __init__(self, n_actions, input_shape, lr):
        super().__init__()
        h, w, c = input_shape  # Height, Width, Channels

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = 64 * h * w
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = state.permute(0, 3, 1, 2).float()  # [B,H,W,C] -> [B,C,H,W]
        return Categorical(self.fc(self.conv(x)))

    def save_model(self):
        save_checkpoint(self, self.optimizer, 'PPO/Actor.pth')

    def load_model(self):
        load_checkpoint(self, self.optimizer, 'PPO/Actor.pth', device=self.device)

class Critic(nn.Module):
    def __init__(self, input_shape, lr):
        super().__init__()
        h, w, c = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = 64 * h * w
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = state.permute(0, 3, 1, 2).float()
        return self.fc(self.conv(x)).squeeze(-1)

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
        state = T.tensor(np.array(observation, dtype=np.float32)).unsqueeze(0).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        return T.squeeze(action).item(), T.squeeze(dist.log_prob(action)).item(), T.squeeze(value).item()

    def learn(self):
        states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        values = T.tensor(vals, dtype=T.float, device=self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float, device=self.actor.device)
        dones = T.tensor(dones, dtype=T.float, device=self.actor.device)

        # Compute advantages using GAE
        advantage = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.gamma * values[k+1] * (1 - dones[k]) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage, dtype=T.float, device=self.actor.device)

        for _ in range(self.n_epochs):
            for batch in batches:
                batch_states = T.tensor(states[batch], dtype=T.float, device=self.actor.device)
                batch_actions = T.tensor(actions[batch], device=self.actor.device)
                batch_old_probs = T.tensor(old_probs[batch], dtype=T.float, device=self.actor.device)

                dist = self.actor(batch_states)
                critic_val = self.critic(batch_states)
                critic_val = T.squeeze(critic_val)

                new_probs = dist.log_prob(batch_actions)
                prob_ratio = (new_probs - batch_old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_val) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()

    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
