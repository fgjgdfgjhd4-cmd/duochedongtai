
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to: " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to: cpu")
print("============================================================================================")


################################## PPO Policy ##################################

# ECA_RESNET
class ECA(nn.Module):
    def __init__(self, c, b=1, gamma=2):
        super(ECA, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):  # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):  # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = ECA(planes)  # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:  # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):  # 右侧的 residual block 结构（50-layer、101-layer、152-layer）
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):  # 三层卷积 Conv2d + Shutcuts
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.channel = ECA(self.expansion * planes)  # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:  # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECA_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=16):
        super(ECA_ResNet, self).__init__()
        self.in_planes = 8

        self.conv1 = nn.Conv2d(3, 8, kernel_size=20,
                               stride=2, padding=10, bias=False)  # conv1
        self.bn1 = nn.BatchNorm2d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)  # conv2_x
        # self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)      # conv3_x
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)      # conv4_x
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)      # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    # 没有state_dim是因为第一个层打算用CNN
    def __init__(self, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 大小变成400*400
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 大小变成200*200
                nn.Flatten(),
                nn.Dropout(p=0.1),
                nn.Linear(200 * 200 * 16, 32),
                nn.Relu(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(16, action_dim),
                nn.Tanh()
            )
        else:

            # ECA_RES
            # self.actor = nn.Sequential(
            #     ECA_ResNet(Bottleneck, [3]),
            #     nn.Linear(16, action_dim),
            #     nn.Softmax(dim=-1)
            # )

            """这一个的问题是在第一次更新之后就变得很极端"""
            # self.actor = nn.Sequential(
            #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=25, stride=2),    # 通过两个卷积层来增大感受野,同时减少参数量
            #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=25, stride=2),    # 输出为182*182
            #     nn.BatchNorm2d(8),
            #     nn.LeakyReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=3, stride=2),  # 大小变成90*90
            #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),    # 输出为45*45
            #     nn.BatchNorm2d(16),
            #     nn.LeakyReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=3, stride=2),  # 大小变成21*21
            #     nn.Flatten(),
            #     nn.Dropout(p=0.1),
            #     nn.Linear(21*21*16, 32),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(32, 16),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(p=0.1),
            #     nn.Linear(16, action_dim),
            #     nn.Softmax(dim=-1)
            # )

            self.actor = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=55, stride=5),  # 输出为150*150
                nn.BatchNorm2d(8),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=3),  # 输出为50*50
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=3),  # 输出为16*16
                nn.BatchNorm2d(8),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=4, stride=2),  # 输出为7*7
                nn.Flatten(),
                nn.Dropout(p=0.1),
                nn.Linear(7 * 7 * 8, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(16, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        """这一个的问题是在第一次更新之后就变得很极端"""
        # self.critic = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=25, stride=2),    # 通过两个卷积层来增大感受野,同时减少参数量
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=25, stride=2),    # 输出为182*182
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # 大小变成90*90
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),    # 输出为45*45
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # 大小变成21*21
        #     nn.Flatten(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(21*21*16, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 16),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(16, 1),
        # )

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=55, stride=5),  # 输出为150*150
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 输出为50*50
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=3),  # 输出为16*16
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),  # 输出为7*7
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(7 * 7 * 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(16, 1)
        )

        # ECA_RES
        # self.critic = nn.Sequential(
        #     ECA_ResNet(Bottleneck, [3]),
        #     nn.Linear(16, 1)
        # )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), dist.probs, action_logprob.detach(), state_val.detach()
        # return dist.probs, action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        # ])

        self.optimizer = torch.optim.RMSprop([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        self.policy_old = ActorCritic(action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        # if self.has_continuous_action_space:

        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

        # else:
        #     print("--------------------------------------------------------------------------------------------")
        #     print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        #     print("--------------------------------------------------------------------------------------------")

    def decay_learning_rate(self, current_time_step, total_time_steps):
        training_frac = 1.0 - current_time_step / total_time_steps
        actor_new_lr = max(0.0001, 0.0003 * training_frac)
        critic_new_lr = max(0.0003, 0.001 * training_frac)
        for param_group in self.optimizer.param_groups:
            if 'actor' in param_group['params'][0].__module__:  # 假设可以根据模块名区分 actor 和 critic
                param_group['lr'] = actor_new_lr
            elif 'critic' in param_group['params'][0].__module__:  # 假设可以根据模块名区分 actor 和 critic
                param_group['lr'] = critic_new_lr

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            """因为这里是连续动作空间，先不改"""
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action_for_train, probabilities, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action_for_train)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return probabilities.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                # state = torch.no_grad()
                action_for_train, probabilities, action_logprob, state_val = self.policy_old.act(state.to(device))

            self.buffer.states.append(state)
            self.buffer.actions.append(action_for_train)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return probabilities

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        print(old_states.shape)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

