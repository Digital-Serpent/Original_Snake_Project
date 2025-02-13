#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size,folder, filename):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU()) # Add ReLU after the first hidden layer


        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU()) # Add ReLU after each subsequent hidden layer

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Create a sequential model from the list of layers
        self.linear = nn.Sequential(*layers)
        self.folder = folder
        self.filename = filename
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def forward(self, x):
        x = self.linear(x)
        return x
    
    def save(self):
        file_name = os.path.join(self.folder, self.filename)
        torch.save(self.state_dict(), file_name)
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self,state,action,reward,next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n,x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                # Q_new += self.gamma*torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new


        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

