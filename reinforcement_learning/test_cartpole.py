import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    model_dir="./models_cartpole"
    num_train_episodes = 500
    

    n_test_episodes = 15
    state_dim = 4
    num_actions = 2

    lr=1e-4
    epsilon_max=0.2
    use_double=True
    #decay="experimental"
    decay="exponential"

    if use_double:
        name = f"cartpole_ne{num_train_episodes}_lr{lr}_ep{epsilon_max}{decay}best_ddqn_agent.pt"
    else:
        name = f"cartpole_ne{num_train_episodes}_lr{lr}_ep{epsilon_max}{decay}best_dqn_agent.pt"
    env = gym.make("CartPole-v0").unwrapped

    # TODO: load DQN agent
    # ...

    Q_head = MLP(state_dim, num_actions)
    Q_tail = MLP(state_dim, num_actions)

    agent = DQNAgent(Q_head, Q_tail, num_actions)
    agent.load(model_dir+"/"+name)


    episode_rewards = []
    for i in range(n_test_episodes):
        stats, loss = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  
    name = f"final_{str(n_test_episodes)}_trails_"+name

    decay = "exponential" if decay=="" else decay
    if use_double:
        fname = "./results/cartpole_results_ddqn-%s_%s.json" % (datetime.now().strftime("%Y%m%d-%H%M%S"), name)
    else:
        fname = "./results/cartpole_results_dqn-%s_%s.json" % (datetime.now().strftime("%Y%m%d-%H%M%S"), name)
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

