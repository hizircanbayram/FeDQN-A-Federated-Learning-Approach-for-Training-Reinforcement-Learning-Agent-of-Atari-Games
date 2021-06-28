import cv2
import gym
import numpy as np

from collections import deque
from gym import spaces

import time

import gym
import gym.spaces
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import os
import datetime
import json

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class ARGS():
    def __init__(self):
        self.env_name = 'PongDeterministic-v4'
        self.render = False
        self.episodes = 1500
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_final=0.02
        self.seed = 1773
        
        self.use_gpu = torch.cuda.is_available()
        
        self.mode = ["rl", "fl_normal"][1]
        
        self.number_of_samples = 5 if self.mode != "rl" else 1
        self.fraction = 0.4 if self.mode != "rl" else 1
        self.local_steps = 50 if self.mode != "rl" else 100
        self.rounds = 25 if self.mode != "rl" else 25
        
        
        self.max_epsilon_steps = self.local_steps*200
        self.sync_target_net_freq = self.max_epsilon_steps // 10
        
        self.folder_name = f"runs/{self.mode}/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")
        
        self.replay_buffer_fill_len = 100
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
args = ARGS()
set_seed(args.seed)

device = torch.device("cuda:0")
dtype = torch.float

os.makedirs('runs/', exist_ok=True)
os.makedirs(f'runs/{args.mode}/', exist_ok=True)
os.makedirs(args.folder_name, exist_ok=True)

# save the hyperparameters in a file
f = open(f'{args.folder_name}/args.txt', 'w')
for i in args.__dict__:
    f.write(f'{i}: {args.__dict__[i]}\n')
f.close()


# utils for atari

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


	
class ReplayMemory():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        if self.count() < batch_size:
            batch = random.sample(self.buffer, self.count())
        else:
            batch = random.sample(self.buffer, batch_size)
            
        state_batch = np.array([np.array(experience[0]) for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        done_batch = np.array([experience[3] for experience in batch])
        next_state_batch = np.array([np.array(experience[4]) for experience in batch])
        
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def count(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out



class PongAgent:
    def __init__(self, args, name = ""):
        self.env = make_env(args.env_name)
        self.num_actions = self.env.action_space.n
        
        self.args = args
        self.name = name
        
        self.dqn = DQN(self.num_actions)
        self.target_dqn = DQN(self.num_actions)
        
        if args.use_gpu:
            self.dqn.cuda()
            self.target_dqn.cuda()        
        
        self.buffer = ReplayMemory(1000000 // 4)
        
        self.gamma = 0.99
        
        self.mse_loss = nn.MSELoss()
        self.optim = optim.RMSprop(self.dqn.parameters(), lr=0.0001)
        
        self.out_dir = './model'
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        
    def to_var(self, x):
        x_var = Variable(x)
        if self.args.use_gpu:
            x_var = x_var.cuda()
        return x_var

        
    def predict_q_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.dqn(states)
        return actions

    
    def predict_q_target_values(self, states):
        states = self.to_var(torch.from_numpy(states).float())
        actions = self.target_dqn(states)
        return actions

    
    def select_action(self, state, epsilon):
        choice = np.random.choice([0, 1], p=(epsilon, (1 - epsilon)))
        
        if choice == 0:
            return np.random.choice(range(self.num_actions))
        else:
            state = np.expand_dims(state, 0)
            actions = self.predict_q_values(state)
            return np.argmax(actions.data.cpu().numpy())

        
    def update(self, states, targets, actions):
        targets = self.to_var(torch.unsqueeze(torch.from_numpy(targets).float(), -1))
        actions = self.to_var(torch.unsqueeze(torch.from_numpy(actions).long(), -1))
        
        predicted_values = self.predict_q_values(states)
        affected_values = torch.gather(predicted_values, 1, actions)
        loss = self.mse_loss(affected_values, targets)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        
    def get_epsilon(self, total_steps, max_epsilon_steps, epsilon_start, epsilon_final):
        return max(epsilon_final, epsilon_start - total_steps / max_epsilon_steps)

    
    def sync_target_network(self):
        primary_params = list(self.dqn.parameters())
        target_params = list(self.target_dqn.parameters())
        for i in range(0, len(primary_params)):
            target_params[i].data[:] = primary_params[i].data[:]
            
            
    def calculate_q_targets(self, next_states, rewards, dones):
        dones_mask = (dones == 1)
        
        predicted_q_target_values = self.predict_q_target_values(next_states)
        
        next_max_q_values = np.max(predicted_q_target_values.data.cpu().numpy(), axis=1)
        next_max_q_values[dones_mask] = 0 # no next max Q values if the game is over
        q_targets = rewards + self.gamma * next_max_q_values
        
        return q_targets
    
        
    def load_model(self, filename):
        self.dqn.load_state_dict(torch.load(filename))
        self.sync_target_network()
        
        
    def play(self, episodes):
        rewards = []
        for i in range(1, episodes + 1):
            done = False
            state = self.env.reset()
            total_reward = 0
            while not done:
                action = self.select_action(state, 0) # force to choose an action from the network
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                # self.env.render()
                
            rewards.append(total_reward)
            
        return rewards
            
                
    def close_env(self):
        self.env.close()
        
        
    def train(self, replay_buffer_fill_len, batch_size, episodes,
              max_epsilon_steps, epsilon_start, epsilon_final, sync_target_net_freq):


        rewards = []
        running_rewards = []
        total_steps = 0
        running_episode_reward = 0
        
        state = self.env.reset()
        for i in range(replay_buffer_fill_len):
            action = self.select_action(state, 1) # force to choose a random action
            next_state, reward, done, _ = self.env.step(action)
            
            self.buffer.add(state, action, reward, done, next_state)
            
            state = next_state
            if done:
                self.env.reset()

                
        # main loop - iterate over episodes
        for i in range(1, episodes + 1):
            # reset the environment
            done = False
            state = self.env.reset()
            
            # reset spisode reward and length
            episode_reward = 0
            episode_length = 0
            
            # play until it is possible
            while not done:
                # synchronize target network with estimation network in required frequence
                if (total_steps % sync_target_net_freq) == 0:
                    self.sync_target_network()

                # calculate epsilon and select greedy action
                epsilon = self.get_epsilon(total_steps, max_epsilon_steps, epsilon_start, epsilon_final)
                action = self.select_action(state, epsilon)
                
                # execute action in the environment
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add(state, action, reward, done, next_state)
                
                # sample random minibatch of transactions
                s_batch, a_batch, r_batch, d_batch, next_s_batch = self.buffer.sample(batch_size)
                
                # estimate Q value using the target network
                q_targets = self.calculate_q_targets(next_s_batch, r_batch, d_batch)
                
                # update weights in the estimation network
                self.update(s_batch, q_targets, a_batch)
                
                # set the state for the next action selction and update counters and reward
                state = next_state
                total_steps += 1
                episode_length += 1
                episode_reward += reward
                
            running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward
            
            running_rewards.append(running_episode_reward)
            rewards.append(episode_reward)

            
        return rewards, running_rewards

class FederatedLearning:
    
    def __init__(self, args):
        self.args = args
        self.main_agent = PongAgent(args, "main")
        
        self.create_clients()
        
        self.logs = {}
        
    def create_clients(self):
        self.clients = {}
        self.client_names = []
        self.updated_clients = {}
        for i in range(self.args.number_of_samples):
            self.client_names.append(f"client_{i}")
            self.clients[f"client_{i}"] = PongAgent(args, i)
            self.updated_clients[f"client_{i}"] = 0
            
            
    def update_clients(self):
        with torch.no_grad():
            def update(client_layer, main_layer):
                client_layer.weight.data = main_layer.weight.data.clone()
                client_layer.bias.data = main_layer.bias.data.clone()
                
            for i in range(self.args.number_of_samples):
                update(self.clients[self.client_names[i]].dqn.conv1, self.main_agent.dqn.conv1)
                update(self.clients[self.client_names[i]].dqn.conv2, self.main_agent.dqn.conv2)
                update(self.clients[self.client_names[i]].dqn.conv3, self.main_agent.dqn.conv3)
                
                
                update(self.clients[self.client_names[i]].dqn.fc1, self.main_agent.dqn.fc1)
                update(self.clients[self.client_names[i]].dqn.fc2, self.main_agent.dqn.fc2)
                

                del self.clients[self.client_names[i]].buffer
                self.clients[self.client_names[i]].buffer = ReplayMemory(1000000 // 4)

                del self.clients[self.client_names[i]].target_dqn
                self.clients[self.client_names[i]].target_dqn = DQN(self.main_agent.num_actions)
                self.clients[self.client_names[i]].target_dqn.load_state_dict(self.clients[self.client_names[i]].dqn.state_dict()) 
                
                if self.args.use_gpu:
                    self.clients[self.client_names[i]].target_dqn.cuda()     



    def update_main_agent(self, round_no):
        # meaning
        conv1_mean_weight = torch.zeros(size=self.main_agent.dqn.conv1.weight.shape).to(device)
        conv1_mean_bias = torch.zeros(size=self.main_agent.dqn.conv1.bias.shape).to(device)

        conv2_mean_weight = torch.zeros(size=self.main_agent.dqn.conv2.weight.shape).to(device)
        conv2_mean_bias = torch.zeros(size=self.main_agent.dqn.conv2.bias.shape).to(device)

        conv3_mean_weight = torch.zeros(size=self.main_agent.dqn.conv3.weight.shape).to(device)
        conv3_mean_bias = torch.zeros(size=self.main_agent.dqn.conv3.bias.shape).to(device)

        linear1_mean_weight = torch.zeros(size=self.main_agent.dqn.fc1.weight.shape).to(device)
        linear1_mean_bias = torch.zeros(size=self.main_agent.dqn.fc1.bias.shape).to(device)

        linear2_mean_weight = torch.zeros(size=self.main_agent.dqn.fc2.weight.shape).to(device)
        linear2_mean_bias = torch.zeros(size=self.main_agent.dqn.fc2.bias.shape).to(device)
        
        number_of_samples = self.args.number_of_samples
        with torch.no_grad():

            for i in range(number_of_samples):
                conv1_mean_weight += self.clients[self.client_names[i]].dqn.conv1.weight.clone()
                conv1_mean_bias += self.clients[self.client_names[i]].dqn.conv1.bias.clone()

                conv2_mean_weight += self.clients[self.client_names[i]].dqn.conv2.weight.clone()
                conv2_mean_bias += self.clients[self.client_names[i]].dqn.conv2.bias.clone()

                conv3_mean_weight += self.clients[self.client_names[i]].dqn.conv3.weight.clone()
                conv3_mean_bias += self.clients[self.client_names[i]].dqn.conv3.bias.clone()

                linear1_mean_weight += self.clients[self.client_names[i]].dqn.fc1.weight.clone()
                linear1_mean_bias += self.clients[self.client_names[i]].dqn.fc1.bias.clone()

                linear2_mean_weight += self.clients[self.client_names[i]].dqn.fc2.weight.clone()
                linear2_mean_bias += self.clients[self.client_names[i]].dqn.fc2.bias.clone()

                
            conv1_mean_weight = conv1_mean_weight / number_of_samples
            conv1_mean_bias = conv1_mean_bias / number_of_samples

            conv2_mean_weight = conv2_mean_weight / number_of_samples
            conv2_mean_bias = conv2_mean_bias / number_of_samples

            conv3_mean_weight = conv3_mean_weight / number_of_samples
            conv3_mean_bias = conv3_mean_bias / number_of_samples

            linear1_mean_weight = linear1_mean_weight / number_of_samples
            linear1_mean_bias = linear1_mean_bias / number_of_samples

            linear2_mean_weight = linear2_mean_weight / number_of_samples
            linear2_mean_bias = linear2_mean_bias / number_of_samples
            
            
            with torch.no_grad():
                def update(main_layer, averaged_layer_weight, averaged_layer_bias):
                    main_layer.weight.data = averaged_layer_weight.data.clone()
                    main_layer.bias.data = averaged_layer_bias.data.clone()
                
                update(self.main_agent.dqn.conv1, conv1_mean_weight, conv1_mean_bias)
                update(self.main_agent.dqn.conv2, conv2_mean_weight, conv2_mean_bias)
                update(self.main_agent.dqn.conv3, conv3_mean_weight, conv3_mean_bias)
                
                
                update(self.main_agent.dqn.fc1, linear1_mean_weight, linear1_mean_bias)
                update(self.main_agent.dqn.fc2, linear2_mean_weight, linear2_mean_bias)
            

        
        
    def step(self, idx_users, round_no):
        
        self.update_clients()
        
        for user in idx_users:
            print(f"Client {user}")
            
            rewards, running_rewards = self.clients[self.client_names[user]].train(
                replay_buffer_fill_len = self.args.replay_buffer_fill_len, 
                batch_size = self.args.batch_size, 
                episodes = self.args.local_steps,
                max_epsilon_steps = self.args.max_epsilon_steps,
                epsilon_start = self.args.epsilon_start - 0.03*(round_no - 1),
                epsilon_final = self.args.epsilon_final,
                sync_target_net_freq = self.args.sync_target_net_freq)
            
            print(f'LOCAL TRAIN: Avg Reward: {np.array(rewards).mean():.5f},  Avg Running Reward: {np.array(running_rewards).mean():.5f}')
            

            self.logs[f"{round_no}"]["train"]["rewards"].append(rewards)
            self.logs[f"{round_no}"]["train"]["running_rewards"].append(running_rewards)
            
        self.update_main_agent(round_no)
            
        self.logs[f"{round_no}"]["eval"]["rewards"] = self.main_agent.play(10)
        
        
        
    def run(self):
        
        m = max(int(self.args.fraction * self.args.number_of_samples), 1) 
        for round_no in range(self.args.rounds):
            
            self.logs[f"{round_no + 1}"] = {"train": {
                                            "rewards": [],
                                            "running_rewards": []
                                        },
                                        "eval": {
                                            "rewards": None
                                        }
                                       }
            idxs_users = np.random.choice(range(self.args.number_of_samples), m, replace=False)

            for user in idxs_users:
                self.updated_clients[f"client_{user}"] = round_no + 1
                
            self.step(idxs_users, round_no+1)
            print(f'{round_no + 1}/{self.args.rounds}')
            print(f'TRAIN: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["rewards"]).mean():.5f},  Avg Running Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["running_rewards"]).mean():.5f}')
            print(f'EVAL: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["eval"]["rewards"]).mean():.5f}')

        
        with open(args.folder_name + "/train.txt", 'w') as convert_file:
             convert_file.write(json.dumps(self.logs))
                
        torch.save(self.main_agent.dqn.state_dict(), f'{args.folder_name}/model.pt')


fl = FederatedLearning(args)
fl.run()