import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_num)
        )

    def forward(self, x):
        x = self.fc(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        self.model_name = args.model_name
        self.log_name = args.log_name

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # saved rewards and actions
        self.rewards, self.log_probs = [], []


    def save(self, save_path):
        save_path += '.cpt'
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.log_probs = [], []

    def make_action(self, state, test=False):
        #action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        action_prob = self.model(torch.from_numpy(state).unsqueeze(0))
        log_prob = action_prob.log()
        try:
            action = Categorical(action_prob).sample()[0].numpy()
        except:
            logging.info('error: {}'.format(action_prob))
            action = self.env.action_space.sample()

        if test:
            return action
        else:
            return action, log_prob[0][action]

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        R = torch.zeros(1)
        loss = 0
        for reward, log_prob in zip(self.rewards[::-1], self.log_probs[::-1]):
            R = self.gamma * R + reward
            loss = loss - (log_prob * R)
        loss /= len(self.rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(message)s', 
            handlers=[logging.FileHandler(self.log_name, 'w'), logging.StreamHandler(sys.stdout)]
        )

        best_reward = -100000000
        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):    
                action, log_prob = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.log_probs.append(log_prob)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                logging.info('Epochs: %d/%d | Avg reward: %f | Best reward: %f'%
                       (epoch, self.num_episodes, avg_reward, best_reward))

            if avg_reward > best_reward: # to pass baseline, avg. reward > 50 is enough.
                print('upd best reward: {} -> {}'.format(best_reward, avg_reward))
                best_reward = avg_reward
                self.save(self.model_name)
