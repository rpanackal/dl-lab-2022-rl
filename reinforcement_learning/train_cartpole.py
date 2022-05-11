from cmath import e, log
import sys
sys.path.append("../") 

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats
import time

def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()
    loss = 0
    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)

        #continue
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            loss=agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

        

    return stats, loss

def train_online(env, agent, num_episodes, eval_cycle, num_eval_episodes, 
    model_dir="./models_cartpole", tensorboard_dir="./tensorboard",
    epsilon_max=0.9, lr=1e-5, use_double=False, decay="exponential"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    name = "cartpole_ne"+str(num_episodes)+"_lr"+str(lr)+"_ep"+str(epsilon_max)+decay

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name=name,
    stats=["episode_reward", "a_0", "a_1", "epsilon", "loss", "use_double"])
    tensorboard_val = Evaluation(os.path.join(tensorboard_dir, "val"), name=name,
    stats=["episode_reward", "a_0", "a_1", "epsilon", "loss", "use_double"])

    # training
    best_val = 0
    for i in range(num_episodes):
        print("episode: ",i)
        stats, loss = run_episode(env, agent, deterministic=False, do_training=True, rendering=True)
        tensorboard.write_episode_data(i+1, {"episode_reward" : stats.episode_reward,
                                                "a_0" : stats.get_action_usage(0),
                                                "a_1" : stats.get_action_usage(1),
                                                "epsilon": agent.epsilon, 
                                                "loss": loss,
                                                "use_double": use_double})

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats_val, loss_val = run_episode(env, agent, deterministic=True, do_training=False)
                tensorboard_val.write_episode_data(i, eval_dict={  "episode_reward" : stats_val.episode_reward, 
                                                                "a_0" : stats_val.get_action_usage(0),
                                                                "a_1" : stats_val.get_action_usage(1),
                                                                "epsilon": agent.epsilon, 
                                                                "loss": loss_val,
                                                                "use_double": use_double})
        
        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            if use_double:
                agent.save(os.path.join(model_dir, name+"ddqn_agent.pt"))
            else:
                agent.save(os.path.join(model_dir, name+"dqn_agent.pt"))
        if stats_val.episode_reward > best_val:
            if use_double:
                agent.save(os.path.join(model_dir, name+"best_ddqn_agent.pt"))
            else:
                agent.save(os.path.join(model_dir, name+"best_dqn_agent.pt"))


        
        # b = (i+1)/num_episodes
        # agent.epsilon = (1-b)*epsilon_max + b*0.05
        if decay=="experimental":
            agent.epsilon = ((e**(1 - (i+1)/num_episodes))/e)*epsilon_max
        elif decay=="linear":
            agent.epsilon = (1 - (i+1)/num_episodes)*epsilon_max
        elif decay=="exponential":
            agent.epsilon = agent.epsilon * 0.95
        else:
            continue
   
    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 10         # evaluate every 10 episodes
    num_episodes=500

    gamma=0.95
    batch_size=64
    epsilon_max = 0.2
    tau=0.01
    lr=1e-4
    history_length=0
    capacity=1e5
    use_double= True
    decay="exponential"

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    Q_head = MLP(state_dim, num_actions)
    Q_tail = MLP(state_dim, num_actions)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(Q_head, Q_tail, num_actions, lr=lr, epsilon=epsilon_max, use_double=use_double)

    
    # 3. train DQN agent with train_online(...)
    # What is number of episodes ?
    train_online(env, agent, num_episodes=num_episodes, eval_cycle=eval_cycle, 
        num_eval_episodes=num_eval_episodes, epsilon_max=epsilon_max, lr=lr, 
        use_double=use_double, decay=decay)
 
