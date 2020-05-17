import random
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import logging

from agent_dir.agent import Agent
from environment import Environment
from collections import deque, namedtuple

use_cuda = torch.cuda.is_available()

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn_gamma_boxing_0.99')

        # discounted reward
        self.GAMMA = args.dqn_gamma

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 10000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 10000 # frequency to save the model
        self.target_update_freq = args.dqn_target_update_freq # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer
        self.model_name = args.model_name
        self.log_name = args.log_name

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.memory = deque(maxlen=self.buffer_size)


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        if test:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            epsilon = 1.0
        else:
            epsilon = 0.9 + (math.exp(self.steps / self.num_timesteps) / math.e * 0.1)
        state = state.to('cuda' if use_cuda else 'cpu')

        if random.random() < epsilon:
            self.online_net.eval()
            with torch.no_grad():
                action = int(self.online_net(state).max(1)[1][0].cpu().numpy())
        else:
            action = self.env.action_space.sample()

        return action

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        
        # step 0
        self.online_net.train()
        self.target_net.eval()

        # step 1
        minibatch = random.sample(self.memory, self.batch_size)
        minibatch = Transition(*zip(*minibatch))
        mask = torch.tensor(
            tuple(map(lambda x: 1 - x, minibatch.done)), 
            device='cuda' if use_cuda else 'cpu',
            dtype=torch.bool
        )
        action = torch.tensor(
            minibatch.action,
            device='cuda' if use_cuda else 'cpu',
        ).unsqueeze(1)
        reward = torch.tensor(
            minibatch.reward,
            device='cuda' if use_cuda else 'cpu',
        ).unsqueeze(1)
        state = torch.cat(minibatch.state).to('cuda' if use_cuda else 'cpu')
        next_state = torch.cat([x for (x, y) in zip(minibatch.next_state, minibatch.done) if not y]).to('cuda' if use_cuda else 'cpu')

        # step 2
        state_action_value = self.online_net(state).gather(1, action)

        # step 3
        with torch.no_grad():
            next_state_value = torch.zeros((self.batch_size, 1), device='cuda' if use_cuda else 'cpu')
            next_state_value[mask] = self.target_net(next_state).max(1)[0].unsqueeze(1)
        
        # step 4
        expected_state_action_value = next_state_value * self.GAMMA + reward
        
        # step 5
        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(message)s', 
            handlers=[logging.FileHandler(self.log_name, 'w'), logging.StreamHandler(sys.stdout)]
        )

        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        best_avg_reward = 0
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)

                # TODO: store the transition in memory
                self.memory.append((state, action, next_state, reward, done))
                
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                logging.info('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f | Best: %f'%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss, best_avg_reward))
                if total_reward / self.display_freq > best_avg_reward:
                    best_avg_reward = total_reward / self.display_freq
                    self.save(self.model_name)
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
