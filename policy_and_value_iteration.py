# Spring 2021, IOC 5269 Reinforcement Learning
# HW1, partI: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env:
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_states)])
    
    ##### FINISH TODOS HERE #####
    #Initialize the value function V(s)
    Vs0 = np.zeros(num_states)
    Vs1 = np.zeros(num_states)
    
    #implement value iteration
    for i in range(max_iterations):
        Q=np.zeros((num_states,num_actions))
        #update V(s) one time
        for state in range(num_states):
            for action in range(num_actions):
                #Get transition probabilities and reward function from the gym env
                #though there is actually one next_state, we still write a for loop to scan all the possible next_state without loss of generality
                for prob_transi, next_state, reward, done in env.P[state][action]:
                    if done:
                        Q[state][action]+=prob_transi * reward
                    else:
                        Q[state][action]+=prob_transi * (reward + gamma * Vs0[next_state])  #Q contains the value taking each of the actions
                optimal_action=np.argmax(Q[state])
                #Iterate and improve V(s) using the Bellman optimality operator
                Vs1[state]=Q[state][optimal_action]
        #judge whether V(s) converges i.e. Vs0 equals to Vs1
        if abs(sum(Vs0)-sum(Vs1))<eps:
            break
        else:
            Vs0=Vs1.copy()
    #retrieve the optimal policy from Q
    policy=np.argmax(Q, axis=1)
    #############################
    
    # Return optimal policy
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env:
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_states)])
    
    ##### FINISH TODOS HERE #####
    #policy iteration
    policy = np.ones([num_states, num_actions]) / num_actions #initialize with a policy containing the probability
    corsV = np.zeros(num_states)#initialize with a corresponding V
    for i in range(max_iterations):
        #iterative policy evaluation
        while True:
            corsV_pre=corsV.copy()
            for state in range(num_states):
                v=0
                for action,prob_act in enumerate(policy[state]): #enumerate return corresponding index(=action) and value(=probability of taking this action)
                    for prob_transi, next_state, reward, _ in env.P[state][action]: #consider all the next state
                        v+=prob_act*prob_transi*(reward+gamma*corsV[next_state]) #v is the expected value amoung all the actions
                corsV[state]=v
            if(abs(sum(corsV_pre)-sum(corsV))<eps):
                break
    
        #one-step greedy policy improvement
        policy_stable=True
        for state in range(num_states):
            #given corresponding V, compute equivelent Q
            Qs=np.zeros(num_actions)
            for action in range(num_actions):
                for prob_transi, next_state, reward, _ in env.P[state][action]:
                    Qs[action]+=prob_transi*(reward+gamma*corsV[next_state])
            #derive the new policy
            optimal_action=np.argmax(Qs)
            if(np.any(policy[state] != np.eye(num_actions)[optimal_action])):
               policy_stable=False #when there is any one policy[state] which is not stable, policy_stable=False
            policy[state]=np.eye(num_actions)[optimal_action] #except that optimal action equals to 1, the left element are all equal to 0
        if policy_stable == True:
            break
    #retrieve the optimal policy from Q
    policy=np.argmax(policy,axis=1)
    #############################
    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """
        Enforce policy iteration and value iteration
    """
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc) # env.render()
    
    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')
    
    # For debugging
    #help(env.unwrapped)
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    #print(pi_policy==vi_policy)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])
    print('Discrepancy:', diff)
    



