## DRL - MADDPG Algorithm - Tennis Enviroment. 

## State and Action Spaces
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
<br>
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Learning Algorithm

The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." 
They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks

For the DDPG foundation, I used [this vanilla, single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template. 
Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented a variation of the actor-critic method which was discuss in the class session which is outlined in this paper ([Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf))

The MADDPG agent is contained in [`maddpg_agent.py`](https://github.com/sand47/DRLND-Collaboration-Competion/blob/master/maddpg_agents.py)

### MADDPG Hyper Parameters
- n_episodes (int)      : maximum number of training episodes
- max_t (int)           : maximum number of timesteps per episode
- train_mode (bool)     : if 'True' set environment to training mode

Where
`n_episodes=2000`, `max_t=1000`,'train_mode=True'

### MADDPG Agent Hyper Parameters

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 8e-3              # for soft update of target parameters
- LR_ACTOR = 1e-3        # learning rate of the actor
- LR_CRITIC = 1e-3        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay
- LEARN_EVERY = 1        # learning timestep interval
- LEARN_NUM = 5          # number of learning passes
- OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
- OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
- EPSILON = 1.0           # explore->exploit noise process added to act step
- EPSILON_DECAY = 1e-6    # decay rate for noise process
- SOLVED_SCORE = 0.5
- CONSEC_EPISODES = 100
- PRINT_EVERY = 10
- ADD_NOISE = True

### Neural Networks

Actor and Critic network models were defined in [`ddpg_model.py`](https://github.com/hortovanyi/DRLND-Continuous-Control/blob/master/ddpg_model.py).

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. The network has an initial dimension the same as the state size.

The Critic networks utilised two fully connected layers with 256 and 128 units with leaky_relu activation. The critic network has  an initial dimension the size of the state size plus action size.

## Plot of rewards
![Reward Plot](https://github.com/sand47/DRLND-Collaboration-Competion/blob/master/images/score_eposide.PNG)

```
Episodes 1150	Max Reward: 0.400	Moving Average: 0.198
Episodes 1160	Max Reward: 0.390	Moving Average: 0.204
Episodes 1170	Max Reward: 0.500	Moving Average: 0.200
Episodes 1180	Max Reward: 0.400	Moving Average: 0.195
Episodes 1190	Max Reward: 0.400	Moving Average: 0.209
Episodes 1200	Max Reward: 0.300	Moving Average: 0.203
Episodes 1210	Max Reward: 0.900	Moving Average: 0.207
Episodes 1220	Max Reward: 1.000	Moving Average: 0.204
Episodes 1230	Max Reward: 1.200	Moving Average: 0.220
Episodes 1240	Max Reward: 1.400	Moving Average: 0.238
Episodes 1250	Max Reward: 1.500	Moving Average: 0.272
Episodes 1260	Max Reward: 2.800	Moving Average: 0.317
Episodes 1270	Max Reward: 1.000	Moving Average: 0.353
Episodes 1280	Max Reward: 5.200	Moving Average: 0.435
Episodes 1290	Max Reward: 4.400	Moving Average: 0.496
<-- Environment solved in 1191 episodes!                 
<-- Moving Average: 0.502 over past 100 episodes
Episodes 1300	Max Reward: 0.700	Moving Average: 0.519
Episodes 1310	Max Reward: 3.800	Moving Average: 0.633
Episodes 1320	Max Reward: 1.500	Moving Average: 0.663
Episodes 1330	Max Reward: 0.400	Moving Average: 0.647
Episodes 1340	Max Reward: 0.600	Moving Average: 0.637
Episodes 1350	Max Reward: 3.000	Moving Average: 0.667
Episodes 1360	Max Reward: 1.400	Moving Average: 0.643
Episodes 1370	Max Reward: 1.400	Moving Average: 0.618
Episodes 1380	Max Reward: 1.500	Moving Average: 0.576
Episodes 1390	Max Reward: 1.300	Moving Average: 0.535
Episodes 1400	Max Reward: 1.500	Moving Average: 0.527
Episodes 1410	Max Reward: 1.500	Moving Average: 0.420
Episodes 1420	Max Reward: 0.900	Moving Average: 0.394
Episodes 1430	Max Reward: 0.800	Moving Average: 0.403
Episodes 1440	Max Reward: 1.800	Moving Average: 0.434
Episodes 1450	Max Reward: 1.990	Moving Average: 0.419
Episodes 1460	Max Reward: 1.100	Moving Average: 0.443
Episodes 1470	Max Reward: 1.800	Moving Average: 0.489
<-- Best episode so far!                
Episode 1472	Max Reward: 5.200	Moving Average: 0.528
<-- Training stopped. Best score not matched or exceeded for 200 episodes

Environment SOLVED in 7 episode	Moving Average =30.3 over last 100 episodes
```
## Ideas for Future Work

- In order to improve the model, we can try implementing one of the collaboration and competition algorithm mentioned in this link https://github.com/LantaoYu/MARL-Papers
- We can try using dropout,batch_norm and have prioritized experience replay rather than picking at random 
- We can play with the noise and other hyperparameters but I got a better output with the above values. 
