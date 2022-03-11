# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


env = gym.make("Blackjack-v0")
    

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """
    
    # value function
    V = defaultdict(float)
    
    ##### FINISH TODOS HERE #####
    
    #Initialize the value function
    V = defaultdict(float)
    N = defaultdict(int)
    #G = defaultdict(float)
    for i in range(num_episodes):
        #Sample an episode: in each episode, sample a trajectory that follows a specific policy
        trajectory_i=[]
        playerCurrentState=env.reset() #though env has been reset when it is been declared(constructor did it), it should be reset here again since we don't want each trajectory's initial state is the same; both env._get_obs() and env.reset() returns this environment's current state which contains (player's current sum, dealer's showing card, player have usable ace or not); 
        while True:
            playerAction=policy(playerCurrentState) #because the trajectory is drawn under a specific policy, the action can't be randomly taken(env.action_space.sample()); the policy is a function which is taken as incoming parameter(python can directly deliver a function as parameter!!!)
            playerNextState, reward, done, _ = env.step(playerAction)
            trajectory_i.append((playerCurrentState,playerAction,reward))
            if done:
                #trajectory_i.append((playerCurrentState,_,_)) #we skip terminal state to make code more readable
                break
            playerCurrentState=playerNextState
        
        #Calculate sample returns: in each episode, compute the corresponding sample return G_{i,t} in each step t of the i-th sampled trajectory
        G_i=[None]*len(trajectory_i) #initialize G_{i,t} as an empty len(trajectory_i) elements' list
        G_i[-1]=(gamma**(len(trajectory_i)-1))*trajectory_i[-1][2] #trajectory's third term means reward
        for t in reversed(range(len(trajectory_i)-1)): #each reversed step in the trajectory
            G_i[t]=G_i[t+1]+(gamma**t)*trajectory_i[t][2] #note that G_{i,t} means returns accumulate starting from time t, not ending at time t
        
        #Iterate and update the value function
        visit = set() #initialize each state visit status=none
        for t,((pcs,dsc,pua),_,_) in enumerate(trajectory_i): #each step's state in the trajectory
            if (pcs,dsc,pua) not in visit: #first-visit: if state (pcs,dsc,pua) hasn't been visitted
                N[(pcs,dsc,pua)]+=1
                learningRate=1/N[(pcs,dsc,pua)]
                #G[(pcs,dsc,pua)]+=G_i[t]
                #V[(pcs,dsc,pua)]=G[(pcs,dsc,pua)]*learningRate
                MCerror=G_i[t]-V[(pcs,dsc,pua)]
                V[(pcs,dsc,pua)]+=MCerror*learningRate # this better way don't need to keep G[(pcs,dsc,pua)]
                visit.add((pcs,dsc,pua)) #record state (pcs,dsc,pua) is visitted
    
    #############################
    
    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    # value function
    V = defaultdict(float)
    
    ##### FINISH TODOS HERE #####
    #Initialize the value function
    learningRate=0.5
    V = defaultdict(float)
    #TDerrors = defaultdict(list) #to check the convergence
    for i in range(num_episodes):
        #Sample an episode
        playerCurrentState=env.reset()
        while True:
            #take one step in the environment
            playerAction=policy(playerCurrentState)
            playerNextState, reward, done, _ = env.step(playerAction)
            #Calculate TD errors
            TDtarget=reward+gamma*V[playerNextState]
            TDerror=TDtarget-V[playerCurrentState]
            #TDerrors[(playerCurrentState,playerNextState)].append(TDerror) #to check the convergence
            
            #Iterate and update the value function
            V[playerCurrentState]+=TDerror*learningRate
            
            if done:
                break
            playerCurrentState=playerNextState
    
    #to check convergence    
    #all_series = [list(x)[:50] for x in TDerrors.values()]
    #for series in all_series:
    #    plt.plot(series)
    #############################

    return V

    

def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

if __name__ == '__main__':
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")
    

    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")




