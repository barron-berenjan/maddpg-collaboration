[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Collaboration with MADDPG 

This repository provides the code required to train multiple agent using **Multi-Agent Deep Deterministic Policy Gradients** to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]


## Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The __state space__  has 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and to solve the environment, the agent requires to get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Dependencies and Set-up

To set up your python environment to run the code in this repository, follow the instructions below.

1. Clone this repository

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
  
 3. Clone the [DRLND repository](https://github.com/udacity/deep-reinforcement-learning), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

 4. The environment for this project is based on the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). For this project you will __not need__ to install Unity and a pre-built environment can be downloaded from one of the links below. You need only select the environment that matches your operating system:

	- __Linux__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
	- __Mac OSX__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
	- __Windows (32-bit)__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
	- __Windows (64-bit)__: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

__Note: Unzip the file in the same directory as the notebooks from this repository.__


## Training the Agent

Run `main.py` to train the agents. The code will stop training once it reaches an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


## Watch a Trained Agent in Action

Run `Collaboration with MADDPG (Test).ipynb` to watch pre-trained agents in action!



