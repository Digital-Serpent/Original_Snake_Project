#!/usr/bin/python3
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
import model_multiple_hidden
from plotter import plot
import os
import json
import sys
import csv

MAX_MEMORY = 100_000

class Agent:

    def __init__(self,gamma, folder, filename, BATCH_SIZE = 500, LR=0.002):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = gamma # discount rate, smaller than 1
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # [ states, hidden, output]
        self.model = model_multiple_hidden.Linear_QNet(11, [256], 3, folder, filename)
        self.trainer = model_multiple_hidden.QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Load the trained model if it exists
        try:
            self.model.load_state_dict(
                torch.load(os.path.join(folder, filename)))
            self.model.eval()  # Set the model to evaluation mode
            self.get_action = self.get_action_from_model  # Use the model's get_action
        except FileNotFoundError:
            self.get_action = self.get_action_epsilon_greedy  # Use epsilon-greedy get_action
        
    def get_state(self, game):
        """
        Gets the current state of the game.

        Args:
            game: The SnakeGameAI object.

        Returns:
            A NumPy array representing the state of the game.
        """

        straight_move = game.calculateNewHead([1, 0, 0])[0]
        left_move = game.calculateNewHead([0, 1, 0])[0]
        right_move = game.calculateNewHead([0, 0, 1])[0]

        straight = game.isCollision(straight_move) != 0
        left = game.isCollision(left_move) != 0
        right = game.isCollision(right_move) != 0

        # Check if the snake is trapped at the current position
        # is_trapped = game.is_trapped()

        W = game.direction == Direction.WEST
        E = game.direction == Direction.EAST
        N = game.direction == Direction.NORTH
        S = game.direction == Direction.SOUTH

        # Create the state array
        state = [
            # 3
            straight,
            left,
            right,
            # 4
            W,
            E,
            N,
            S,
            # 4 Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            # 2
        ]

        return np.array(state,dtype=int)
    
    def remember(self, state, action,reward, next_state,done):
        self.memory.append((state, action,reward, next_state,done))

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory,self.BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        self.trainer.train_step(*zip(*mini_sample))
    
    def train_short_memory(self, state, action,reward, next_state,done):
        self.trainer.train_step(state, action,reward, next_state,done)

    def get_action_from_model(self, state):
        # Always use the model
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)  # Get the model's prediction
        move = torch.argmax(prediction).item()  # Choose the best action
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move  # Return the chosen action

    def get_action_epsilon_greedy(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 240 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move
    
def saveSnakeMovements(data, folder1, folder2, outputName):
    """
    Saves the snake's movements to a JSON file and game data to a CSV file.

    Args:
        data: A list of the snake's movements. A list of game data to be written to the CSV file.
        folder1: The first folder in the path.
        folder2: The second folder in the path.
        outputName: The base name for the output files (without extension).
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create the folder if they dont exist
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    
    relPath2 = os.path.join(folder1,folder2)
    if not os.path.exists(relPath2):
        os.makedirs(relPath2)
    
    if outputName.endswith('.csv'):
        # save the csv data
        csvFilePath = os.path.join(folder1,folder2, outputName)
        with open(csvFilePath,'a', newline="") as f:
            writer = csv.writer(f)
            # write the game data to the new row
            writer.writerow(data.split(','))
    elif outputName.endswith('.json'):
        jsonFilePath = os.path.join(folder1,folder2,outputName)
        with open(jsonFilePath,'w') as f:
            json.dump(data, f)

def train(gamma=0.66, BATCH_SIZE = 500, LR = 0.002, activateGameWindow = True,activateGraph = True):
    # Get the directory of the current script
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    # change the current working directory 
    os.chdir(scriptDir)

    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = -1 # initialize

    agent = Agent(gamma, './model',f'model_Gamma{gamma}_MEM{MAX_MEMORY}_BATCH{BATCH_SIZE}_LR{LR}.pth', BATCH_SIZE, LR)

    w = 30 # adjust as wanted
    h = 25 # adjust as wanted
    game = SnakeGameAI(activateGameWindow=activateGameWindow,w=w,h=h)
    csvFile = 'AllDataOut.csv'

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move and get the new state
        reward, done, score, causeOfDeath, movements = game.playStep(final_move)

        state_new = agent.get_state(game)

        # train the short memory
        agent.train_short_memory(state_old,final_move, reward,state_new,done)

        # remember
        agent.remember(state_old,final_move, reward,state_new,done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

                # save the snake's movements to a JSON file
                saveSnakeMovements(movements,'ModelRuns',f'model_Gamma{gamma}_MEM{MAX_MEMORY}_BATCH{BATCH_SIZE}_LR{LR}', f'game{agent.n_games}_score{score}.json')
            elif agent.n_games %10 == 0: # save every n games
                saveSnakeMovements(movements,'ModelRuns',f'model_Gamma{gamma}_MEM{MAX_MEMORY}_BATCH{BATCH_SIZE}_LR{LR}', f'game{agent.n_games}_score{score}.json')
            
            totalScore += score
            meanScore = totalScore / agent.n_games

            # save output to the csv file
            saveSnakeMovements(f'{agent.n_games}, {score}, {record}, {meanScore}, {causeOfDeath}', 'ModelRuns', f'model_Gamma{gamma}_MEM{MAX_MEMORY}_BATCH{BATCH_SIZE}_LR{LR}', csvFile)

            print(gamma,'_',BATCH_SIZE,'_',LR,' Game: ',agent.n_games,'\tScore: ',score,'\tRecord: ',record,'\tMean: ',meanScore,'\t',causeOfDeath)

            if activateGraph:
                plotScores.append(score)
                plotMeanScores.append(meanScore)
                plot(plotScores,plotMeanScores)

            # optional            
            if w*h-1 <= score:
                sys.exit()

if __name__ == '__main__':
    # if no arguments are provided, run train() with default values
    if len(sys.argv) == 1:
        train(gamma=0.66, BATCH_SIZE=1000, LR=0.0001, activateGameWindow=True, activateGraph=True)
    else:
        # assuming inputs are in the correct order
        gammaValue = float(sys.argv[1]) # convert to float
        batchValue = int(sys.argv[2]) # convert to int
        lrValue = float(sys.argv[3]) # convert to float

        train(gamma=gammaValue, BATCH_SIZE=batchValue, LR=lrValue, activateGameWindow=True, activateGraph=True)


    

