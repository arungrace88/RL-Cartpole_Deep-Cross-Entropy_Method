# RL-Cartpole_Deep-Cross-Entropy_Method
Solving Cart pole Reinforcement learning (RL) problem using deep Cross Entropy Method (CEM)

The state vector for this system is a 4-D vector. The action has two states - left and right. The episode terminates if any of the below conditions are satisfied:

 - pole angle is more  than 12 degree from vertical axis
 - cart position is more than +/- 2.4cm from the centre
 
The agent receives a reward of 1 for every step taken. The agent is considered to be successful if the average reward is greater than 190.

The screenshot from the simulation is shown below:

![Cartpole_Simulation](https://user-images.githubusercontent.com/20210669/102926056-48a7d180-448c-11eb-967d-510ab8318344.png)

The average reward and the reward threshold for given percentile (70 in this case) is plotted as shown below:
![CartPole_MeanRewards](https://user-images.githubusercontent.com/20210669/102924371-5740b980-4489-11eb-9fd5-772610842fd9.png)
