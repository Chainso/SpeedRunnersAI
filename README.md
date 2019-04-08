SpeedRunnersAI
================

SpeedRunnersAI learns to play SpeedRunners only using images from the screen. The AI can be trained from your own demonstrations, and using reinforcement learning.

&nbsp;

## Dependencies ##

The following dependencies are all required to run the program. They can be installed using pip or Anaconda.

PyUserInput\
H5PY\
PyTorch\
Numpy\
MSS\
OpenCV

&nbsp;

## Setup ##

Set up your key bindings in the [config.ini](config.ini) file. Model parameters can be altered in [train_model_supervised.py](train_model_supervised), and [train_model_reinforcement.py](train_model_reinforcement.py).

&nbsp;

## Recording Gameplay ##

Run record_gameplay.py and open your SpeedRunners game. Using the keybindings set in [config.ini](config.ini), start and stop recording.

&nbsp;

## Training the Model ##

To train the model only using your recorded samples, run [train_model_supervised.py](train_model_supervised.py). To train the model using reinforcement learning and supervised learning (MAIL algorithm), run [train_model_reinforcement.py](train_model_reinforcement.py). For training using reinforcement learning, make sure to run the game offline.

&nbsp;

## Running the Model ##

To run the model on the game, run [run_model.py](run_model.py) and use the keybindings to start and stop the agent.

