# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
from collections import namedtuple
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

import matplotlib.pyplot as plt

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__() #super: use the parent class' function
        
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete) #whether an object is an instance of a class or of a subclass thereof
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        #env.action_space.sample() #see an example of actions
        #env.observation_space.sample() #see an example of states
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        #actor network and critic network share the same input layer
        self.actor_critic_input_layer = nn.Linear(self.observation_dim, self.hidden_size) #create a linear layer with randomly initialized weight and bias and then store them(weight and bias) to actor_critic_input_layer
        #self.actor_critic_input_layer.weight
        #self.actor_critic_input_layer.bias
        #m=nn.linear(len1, len2)
        #=> m(input) , input with shape of (len3,len1)
        #=> output = input @ m.weight^T + m.bias, output with shape of (len2,len3) #if you want to implement and check this formula, remember to convert them to numpy array
        
        #initialize actor network
        self.actor_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.actor_output_layer = nn.Linear(self.hidden_size, self.action_dim)
        
        #initialize critic network
        self.critic_hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.critic_output_layer = nn.Linear(self.hidden_size, 1)  #value approximate function is a scalar
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        tensor_after_input_layer = self.actor_critic_input_layer(state)
        
        #forward actor network
        actor=F.relu(self.actor_hidden_layer(tensor_after_input_layer)) #it seems that relu function can make the convergence faster...
        actor=F.sigmoid(self.actor_output_layer(actor)) #use sigmoid as activation function because the action is either 0 or 1
        action_prob=actor
        
        #forward critic network
        critic=F.relu(self.critic_hidden_layer(tensor_after_input_layer))
        critic=self.critic_output_layer(critic)
        state_value=critic
        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).float() #convert numpy to tensor
        action_prob, state_value = self.forward(state) #input state into actor and critic network to get the network's output, action probability and value approximation function, based on the forward method
        
        m = Categorical(action_prob)
        action = m.sample()
        #action=torch.multinomial(action_prob, 1, replacement=False) #randomly select one action that follows action probability,action_prob
        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    
    def calculate_loss(self, optimizer, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 

        ########## YOUR CODE HERE (8-15 lines) ##########
        len_trajectory=len(saved_actions) #fetch the lenth of a trajectory
        G = torch.zeros(len_trajectory, 1) #initialize value function true based on MC
        G[-1]=(gamma**(len_trajectory-1))*self.rewards[-1]
        VFA=torch.zeros(len_trajectory, 1, requires_grad=True) #initialize value function approximation based on linear neural network
        #caculate the actor(policy) loss and critic(value) loss
        for step in reversed(range(len_trajectory - 1)):
            
            #record critic(value)'s every two compared element, VFA and G, at each step for calculate MSE as critic(value) loss later
            #MC ~ vanilla value function => G
            VFA[step].detach = saved_actions[step].value #use linear neural network to construct value function approximation
            G[step]=G[step+1]+(gamma**step)*self.rewards[step] #calculate value function true G based on MC
            
            #record actor(policy)'s every loss element at each step
            
            advantage = G[step] - VFA[step] #value function true - value function approximation; baseline = VFA
            policy_losses.append(-saved_actions[step].log_prob * advantage.detach()) #"loss" function => add a negative sign
        
        #calculate loss
        value_losses = nn.MSELoss()(G.detach(),VFA) # critic(value) loss
        policy_losses = torch.stack(policy_losses).mean() # actor(policy) loss
        
        loss = value_losses + policy_losses

        ########## END OF YOUR CODE ##########
        
        return loss
    

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9) #在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次, https://blog.csdn.net/qq_20622615/article/details/83150963
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # initialize the record of loss and reward during training to draw and see the trend(performance)
    loss_lst = []
    reward_lst = []
    
    # run inifinitely many episodes
    for i_episode in count(1): #return a count(function of itertools module) object whose .__next__() method returns consecutive values
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        #sample a trajectory
        for step in range(1, 10000): 
            action = model.select_action(state) #for each step, select an action which is sampled from the actor network
            next_state, reward, done, _ = env.step(action) #take the action and get the observed state
            
            state = next_state #update state
            ep_reward += reward #update accumulate rewards
            model.rewards.append(reward) #record reward
            if done: #determine terminate or not
                break
        
        #update actor(policy) and critic(value) network with backpropagation
        # Uncomment the following line to use learning rate scheduler
        scheduler.step() #scheduler.get_last_lr() 
        optimizer.zero_grad() #initialize gradient to zero(because PyTorch accumulates the gradients on subsequent backward passes, https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        loss = model.calculate_loss(optimizer)
        loss.backward() #PyTorch deposits the gradients of the loss w.r.t. each parameter
        optimizer.step() #to adjust the parameters by the gradients collected in the backward pass.
        #print(len(optimizer.param_groups[0]['params']))
        
        # record the loss and reward during training to draw and see the trend(performance)
        loss_lst.append(loss.item())
        reward_lst.append(ep_reward)
        
        #clear the memory to get ready for next trajectory sampling
        model.clear_memory()
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {:.4f}\t Loss: {:.4f}'.format(i_episode, step, ep_reward, ewma_reward, loss))

        # check if we have "solved" the cart pole problem
        if ewma_reward >= env.spec.reward_threshold: #env.spec.max_episode_steps
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, step))
            break
    
    # after ending the training, let's draw the curve to see the training performance
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.plot(loss_lst)
    plt.savefig("CartPole.png")
    plt.cla()

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(reward_lst)
    plt.savefig("CartPole_reward.png")

def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20  
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test('CartPole_{}.pth'.format(lr))
