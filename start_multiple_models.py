#!/usr/bin/python3
import subprocess
import os
from collections import namedtuple
import platform

# Use a list to define the gamma values you want to test
list1 = []
model = namedtuple('model', 'gamma, batch, lr')

list1.append(model(0.66,1000,0.002))
# list1.append(model(0.9,1000,0.1))
# list1.append(model(0.9,1000,0.01))
list1.append(model(0.9,1000,0.001 ))
# list1.append(model(0.9,1000,0.0001))

# Get the directory of the current script
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory
os.chdir(scriptDir)

folders = ['model','ModelRuns']
# Create the folders if they don't exist
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

if __name__ == '__main__':
    processes = []
    for value in list1:
        # Construct the command to run agent.py with the current gamma value
        # This assumes your venv is named.venv
        command = [f"\"{os.path.join(scriptDir, '.venv', 'Scripts', 'python')}\"",  # Path to venv python
                   "agent.py", str(value.gamma), str(int(value.batch)), str(value.lr)]
        
        systemPlatform = platform.system()

        if systemPlatform == "Windows":
            os.system("start cmd /K " + " ".join(command))
        elif systemPlatform == "Linux":
            subprocess.Popen(['gnome-terminal', '--', *command]) # command for linux
