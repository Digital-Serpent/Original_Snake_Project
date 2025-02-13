#!/usr/bin/python3

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time
import json
import sys, os

SNAKE_NAME = 'Original Snake'

# Get the directory of the current script
scriptDir = os.path.dirname(os.path.abspath(__file__))

# change directory to the script's directory
os.chdir(scriptDir)

# initialize the pygame instance
pygame.init()

# Initialize the mixer module
pygame.mixer.init()

# set the default font and the font size
font = pygame.font.Font("cour.ttf",25)

# set up direction class, using global directions
class Direction(Enum):
    WEST = 1
    EAST = 2
    NORTH = 3
    SOUTH = 4

# set up the Point tuple with x and y coordinate
Point = namedtuple('Point','x ,y')

# setup Colors using RGB (will return to later)
BLACK  = (20,20,20)
GREEN1 = (0,128,0) # dark Green
GREEN2 = (0,255,0) # light green
GREEN3 = (50,205,50) # medium green
RED    = (255,0,0)
YELLOW = (248,222,126)
WHITE  = (255,255,255)


# set the blocksize (pixel size)
BLOCK_SIZE = 30

class SnakeGameAI:
    """
        Variables in the Class:
    
    .w         - (x) width of the game 
    .h         - (y) height of the game
    .newMoves  - Number of moves given to the snake when it has eaten food
    .activateGameWindow - Boolean of whether to display the gaming window or not
    .direction - direction the snake's head is moving (referenced from the game window)
    .head      - head of the snake (should always be equal to self.snake[0])
    .snake     - entire body of the snake. this is a list and should grow as the snake does
    .score     - score of the snake (number of food the snake has eaten)
    .moves     - remaining number of moves before the snake starves
    .consecutiveTurns - Number of similar turns in a row (+ means right, - means left) ( this make the snke more fun and less stick to the edge
    .food      - Point locaiton of the food
    .movements - sotre the width, height, all snake's for each move, and the location of the food (used for replay)
    .display   - the game board pygame
    .replay    - boolean, true is a game is being loaded from a json file
    
    """



    def __init__(self, w = 30, h = 25, activateGameWindow = True, replay = False):
        """
        Description

            Initializes the Class, with defaults for the width, height, and activategamewindow

            calcualtes the number of moves for consumed food

            initializes the display (if required)

        Inputs
            
            w - width of the game
            h - height of the game
            activateGameWindow - Boolean of whether to display the gaming window or not

        """

        # could use later with proximity reward for both x and y: (a*(x-x_food)^2+k)
        # self.a = -0.8
        # self.k = 4

        # Initialize the width, the height, game window Boolean, and the replay
        self.w = w     
        self.h = h   
        self.activateGameWindow = activateGameWindow
        self.replay = replay


        # set the (pixel) value of the text area
        self.textAreaHeight = 100 # adjust as needed

        # calculate the new moves
        self.newMoves = int(0.5*w*h) # adjust as needed

        # set the function that _place_food will use when called (this will be replaced if in 'replay' mode)
        self.placeFood = self.placeFoodActualGame

        # initialize the game window, if true
        if activateGameWindow:
            self.display = pygame.display.set_mode((w * BLOCK_SIZE,h * BLOCK_SIZE+self.textAreaHeight))
            pygame.display.set_caption(SNAKE_NAME)
            
            # set the function that update_ui will use when called
            self.updateUi = self.updateUiWithWindow
            self.uiEvents = self.closeUiIfRequested
            if replay:
                self.placeFood = self.updateUiNoWindow # this function does nothing
        else: # No game window being rendered
            self.updateUi = self.updateUiNoWindow # this function does nothing
            self.uiEvents = self.updateUiNoWindow # this function does nothing

        self.reset()
    
    def reset(self):
        """
        Description
        
            Initializes ro reinitializes the game area, resets the head, snake, score, 
            starting Moves, consectutive turns counters, food
        """

        # initialize the direction of the snake
        self.direction = Direction.EAST

        # initialize the snake in the middle of the game area, with the tail going left
        # assume the game area is large enough
        self.head = Point(int(self.w/2),int(self.h/2))
        self.snake = [self.head]

        for i in range(1,3): # add another two blocks to the snake
            self.snake.append(Point(self.head.x-i, self.head.y))
        
        # initialize the score, moves, and consecutive turns counter
        self.score = 0
        self.moves = 3*self.newMoves # adjust as needed
        self.consecutiveTurns = 0 # (+ means right, - means left)

        # initialize/reset the food
        self.food = None 
        self.placeFood()

        # initialize the movement history (this is used for replaying games later)
        self.movements = {"w": self.w, 
                          "h": self.h,
                          "action": [],
                          "food": [],
                          "reward": [] # not necessary for replay, but good for debugging                          
                          }
    
    def placeFoodActualGame(self):
        """
        Description
        
            places the food in a location in the map, and not in the snake
        """

        while True: # ensure that a valid point is found
            # generate a random x/y locations, need to be in the game area
            # self.w & self.h are outside of the map
            x = random.randint(0,(self.w-1))
            y = random.randint(0,(self.h-1))

            self.food = Point(x,y)

            # check if the position is valid (not in the snake, and inside of the map)
            if not self.snakeHitsSnake(self.food,origin = 0) and not self.hitsBoundary(self.food):
                # exit the while loop once valid
                break
    
    def playStep(self,action):
        """
        Description
        
            accepts the action from outside source and moves the snake, checks for game over,
            places new food, and updates the ui

        Inputs
            
            action - list of 3 possible moves [straight,right,left]
        """

        # check for quit game
        self.uiEvents()

        # move the snake
        self.move(action)

        # add the newhead to the beginning of the snake (tail will be removed later, if necessary)
        self.snake.insert(0,self.head)

        # initialize the reward
        reward = 0
        causeOfDeath = ""
        gameOver = False

        # check for collisions & starvation
        if self.hitsBoundary():
            gameOver = True
            reward = -700 # adjust as needed
            causeOfDeath = 'Hit Wall'
        elif self.snakeHitsSnake():
            gameOver = True
            reward = -1000 # adjust as needed
            causeOfDeath = 'Ate Itself'
        elif self.moves < 0:
            # technically the game is over at zero, but only less than we can simplify this funciton
            # by allowing the fail to occur on the next move (if self.moves == 0 but food is eaten, 
            # then more moves are added and the game continues)
            gameOver = True
            reward = -850 # adjust as needed
            causeOfDeath = 'Starvation'
        
        if gameOver:
            self.updateUi(isDead=True)
            # update the movements dictionary with action and food data ( needs to occur before game over)
            self.updateMovements(action, reward)
            return reward, gameOver, self.score, causeOfDeath, self.movements
        
        # Place the new food or just move

        # check if the snake is 'trapped'
        if self.isTrapped():
            reward -= 150
        
        # new head position is on the food ( dont delete the tail)
        if self.head == self.food:
            # Load the sound effect
            sound_effect = pygame.mixer.Sound("Effects/eating.mp3")

            # Play the sound effect
            sound_effect.play()

            # update the score
            self.score += 1

            # add the reward, want a minimum of value, but a max based off of the moves
            reward += max(400 * self.moves/self.newMoves,80) # adjust as needed

            # place the new food
            self.placeFood()

            # add the moves to the player               (adjust as needed)
            if len(self.snake)/self.newMoves < 0.25:
                self.moves += int(0.5 * self.newMoves)
            elif len(self.snake)/self.newMoves < 0.5:
                self.moves += int(0.75 * self.newMoves)
            else:
                self.moves += int(self.newMoves)
        else: # no food
            # remove the last element in the tail (it has moved)
            self.snake.pop()

            reward += self.calculateTurnPenalty(action)

        self.updateUi(isDead=False)
        # update the movements dictionary with action and food data ( needs to occur after the food is placed)
        self.updateMovements(action, reward)
        return reward, gameOver, self.score, causeOfDeath, None

    def updateMovements(self, action, reward):
        """
        Description
        
            saves the data, made as a function so that it could potentially be turned off during any replay

        Inputs
            
            action - list of 3 possible moves [straight,right,left]
        """
        self.movements['action'].append(action)
        self.movements['food'].append([self.food.x,self.food.y])
        self.movements['reward'].append(reward)
    
    def calculateTurnPenalty(self,action):
        """
        Description
        
            determines if too many of the same turn have been taken in a row

        Inputs
            
            action - list of 3 possible moves [straight,right,left]

        Returns

            reward - negative value if too many turns similar turns occur,otherwise 0
        
        """

        # initialize reward
        reward = 0

        if action[1] == 1: # right turn
            if self.consecutiveTurns < 0: # if left turns
                self.consecutiveTurns = 0 # reset the value
            self.consecutiveTurns += 1
        elif action[2] == 1: # left turn
            if self.consecutiveTurns > 0: # if right turns
                self.consecutiveTurns = 0 # reset the value
            self.consecutiveTurns -= 1
        
        turns = abs(self.consecutiveTurns)
        if turns > 4: # penalize for more than 4 consequtive left turns
            # dont want anything more negative than value
            reward = max(-5 * turns +15, -150) # adjust values as needed
        
        return reward

    def isCollision(self, pt = None):
        """
        Description
        
            Determines if the snake hits a boundary or itself, this is mostly used to pass data to the training agent
        
        Note

            This isn't used in this function, but can be used in the agent script

        Inputs
            
            pt - Point(x,y) to test

        Returns

            numeric value 0 if no, 1 if boundary, 2 if snake
        
        """

        # initialize if no value is given
        if pt is None:
            pt = self.head
        
        # test boundaries
        if self.hitsBoundary(pt):
            return 1
        elif self.snakeHitsSnake(pt):
            return 2
        
        # return nothing
        return 0

    def hitsBoundary(self, pt = None):
        """
        Description
        
            Determines if the point hits a boundary

        Inputs
            
            pt - Point(x,y) to test

        Returns

            boolean, true is snake hits boundary
        
        """
        # initialize if no value is given
        if pt is None:
            pt = self.head

        # if the point is in the game area, then it does not hit the boundary
        if 0 <= pt.x < self.w and 0 <= pt.y < self.h:
            return False
        
        return True

    def snakeHitsSnake(self,pt = None, origin = 1):
        """
        Description
        
            Determines if the point hits the snake

        Inputs
            
            pt - Point(x,y) to test
            origin - should only really be 0 or 1, zero uses the whole snake, 1 uses everything but the head

        Returns

            boolean, true is snake hits boundary


        """
        # initialize if no value is given
        if pt is None:
            pt = self.head
        
        if (pt.x, pt.y) in set((part.x, part.y) for part in self.snake[origin:]):
            return True
        
        return False

    def isTrapped(self):
        """
        Description
        
            Checks if the snake's head is surrounded by its body on all four
            cardinal directions (up, down, left, right).  It doesn't check for
            true trapping (no escape path), but rather if there's a body segment
            directly above, below, to the left, and to the right of the head,
            regardless of distance


        Returns

            boolean, true is the snake is 'trapped'

        """

        head = self.head # Get the position of the snake's head
        body = set((part.x, part.y) for part in self.snake[1:]) # Use a set for efficency

        if any(head.x == x and head.y < y for x,y in body): # above
            if any(head.x == x and head.y > y for x,y in body): # below
                if any(head.x > x and head.y == y for x,y in body): # left
                    if any(head.x < x and head.y == y for x,y in body): # right
                        return True
        
        return False
    
    def closeUiIfRequested(self):
        """
        Description

            Close the pygame instance and stop script if windows is closed
            
        
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
    def updateUiNoWindow(self, isDead = False):
        ''' 
        nothing actually happens here, this splitting was done to avoid constantly checking if
        self.activateGameWindow was false (the value does not change after starting). This should 
        save time whether the game display is active or not

        '''

        ##### NOTE: this funciton is used multiple times, 
        ##### dont add anything here without knowing where its called
        return

    def updateUiWithWindow(self, isDead = False):
        """
        Description
        
            Updates the UI

        Inputs
            
            is_dead - boolean where the sanek is dead (game is over)

        """

        # fill in the back of the display with Black
        self.display.fill(BLACK)

        # draw the snake body
        for pt in self.snake:
            # create the outer box for each point
            pygame.draw.rect(self.display,GREEN1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            # create the inner box (unnecessary, but looks nice)
            pygame.draw.rect(self.display,GREEN2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, BLOCK_SIZE/2, BLOCK_SIZE/2))
        
        
        pygame.draw.rect(self.display,RED if isDead else GREEN3, pygame.Rect(self.head.x * BLOCK_SIZE, self.head.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # draw eyes for the snake
        if self.direction == Direction.WEST:
            pygame.draw.circle(self.display,BLACK,(self.head.x * BLOCK_SIZE + 5 , self.head.y * BLOCK_SIZE + 5),2 )
            pygame.draw.circle(self.display,BLACK,(self.head.x * BLOCK_SIZE + 5 , (self.head.y + 1) * BLOCK_SIZE - 5),2 )
        elif self.direction == Direction.NORTH:
            pygame.draw.circle(self.display,BLACK,(self.head.x * BLOCK_SIZE + 5 , self.head.y * BLOCK_SIZE + 5),2 )
            pygame.draw.circle(self.display,BLACK,((self.head.x + 1) * BLOCK_SIZE - 5 , self.head.y * BLOCK_SIZE + 5),2 )
        elif self.direction == Direction.EAST:
            pygame.draw.circle(self.display,BLACK,((self.head.x + 1) * BLOCK_SIZE - 5 , self.head.y * BLOCK_SIZE + 5),2 )
            pygame.draw.circle(self.display,BLACK,((self.head.x + 1) * BLOCK_SIZE - 5 , (self.head.y + 1) * BLOCK_SIZE - 5),2 )
        elif self.direction == Direction.SOUTH:
            pygame.draw.circle(self.display,BLACK,(self.head.x * BLOCK_SIZE + 5 , (self.head.y + 1) * BLOCK_SIZE - 5),2 )
            pygame.draw.circle(self.display,BLACK,((self.head.x + 1) * BLOCK_SIZE - 5 , (self.head.y + 1) * BLOCK_SIZE - 5),2 )
        
        # Draw the food
        pygame.draw.circle(self.display,YELLOW, (self.food.x * BLOCK_SIZE + BLOCK_SIZE/2,self.food.y * BLOCK_SIZE + BLOCK_SIZE/2), BLOCK_SIZE/2) 


        ##             Text Area

        textAreaColor = (50,50,50) # dark gray color

        # calculate the dimensions of the text area
        textAreaY = self.h * BLOCK_SIZE

        # Draw the text area rectangle
        pygame.draw.rect(self.display, textAreaColor, (0, textAreaY, self.w * BLOCK_SIZE, self.textAreaHeight))

        # display the score
        textScore = font.render(f" Score: {self.score}", True, WHITE)

        # display the number of moves remaining
        textMoves = font.render(f" Moves - {self.moves}", True, WHITE)

        # Calculate the y-coordinate for positioning the text
        textY = textAreaY + 10

        # Render the text
        self.display.blit(textScore, [0, textY]) # position the score at the bottom left (top left of the text area)
        self.display.blit(textMoves, [0, textY+self.textAreaHeight/2]) # position the moves at the bottom left

        pygame.display.flip()

        # pause to be able to see
        if isDead:
            # Load the sound effect
            sound_effect = pygame.mixer.Sound("Effects/DeathSound1.mp3")

            # Play the sound effect
            sound_effect.play()
            time.sleep(2)
        else:
            time.sleep(0.005)

    def move(self,action):
        """
        Description
        
            Moves the snake based on the given action, updates the remaining moves counter

        Inputs
            
            action - list of 3 possible moves [straight,right,left]

        """
        # update the moves counter
        self.moves -= 1

        # uses the calculate new head function and assigns the outputs to self.head and self.direction
        self.head, self.direction = self.calculateNewHead(action)
    
    def calculateNewHead(self,action):
        """
        Description
        
            Calculates the new head position and direction based on the given action. 
            this function can be used when calculating states in the agent

        Inputs
            
            action - list of 3 possible moves [straight,right,left]

        Returns

            tuple containing:
                head - new location of the head based off of the action
                direction - global direction the snake's head is moving (Right, Left, Up, Down)
        
        """

        # clockwise direction of movements
        clockWise = [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH]

        # index of the current direction
        idx = clockWise.index(self.direction)

        if np.array_equal(action, [1,0,0]): # straight
            newDir = self.direction # no change
        elif np.array_equal(action, [0,1,0]): # right
            nextIdx = (idx+1) % 4
            newDir = clockWise[nextIdx]
        elif np.array_equal(action, [0,0,1]): # left
            nextIdx = (idx-1) % 4
            newDir = clockWise[nextIdx]
        else: # something messed up
            print(f"[ERROR] - direction not valid: {action}")
        
        x = self.head.x
        y = self.head.y

        # top left is 0,0
        if newDir == Direction.EAST:
            x += 1
        elif newDir == Direction.WEST:
            x -= 1
        elif newDir == Direction.SOUTH:
            y += 1
        else: # NORTH
            y -= 1
        
        return Point(x,y), newDir


def loadAndReplay(filePath):
    """
        Loads snake movements and food positions from a JSON file and replays the game
    """

    with open(filePath,'r') as f:
        data = json.load(f)
    
    # set up the game, want to show the UI and set the correct width/height
    game = SnakeGameAI(activateGameWindow=True,w = data['w'],h = data['h'], replay = True)

    time.sleep(5)

    game.food = Point(*data['food'][0])
    for action,food in zip(data['action'],data['food']):

        # simulate the action
        game.playStep(action)
        # force a replacement of the food 
        game.food = Point(*food)

        time.sleep(0.005) # adjust as needed
    
    


if __name__ == '__main__':
    loadAndReplay(r"D:\Digital_Serpent\Original_Snake_Project\ModelRuns\model_Gamma0.9_MEM100000_BATCH1000_LR0.001\game840_score10.json")

