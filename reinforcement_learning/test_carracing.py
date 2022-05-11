from __future__ import print_function
import datetime
import json

import gym
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np
import argparse
import os

np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-hl", "--history_length", help="it's in the name", type=int, default=1)
    parser.add_argument("-b", "--batch_size", help="it's in the name", type=int, default=128)
    parser.add_argument("-l", "--learning_rate", help="it's in the name", type=float, default=1e-2)    
    parser.add_argument("-n", "--number_mini_batches", help="it's in the name", type=int, default=1000)
    parser.add_argument("-ec", "--number_eval_cycle", help="it's in the name", type=int, default=10)
    parser.add_argument("-e", "--epsilon", help="decide greedy approach threshold", type=float, default=0.1)
    parser.add_argument("-ne", "--number_episodes", help="it's in the name", type=int, default=15)
    parser.add_argument("-mts", "--max_timesteps", help="it's in the name", type=int, default=1000)
    parser.add_argument("-smts", "--schedule_max_timesteps", help="it's in the name", action="store_true", default=False)
    parser.add_argument("-pre", "--pretrained_weights", help="it's in the name", action="store_true", default=False)
    parser.add_argument("-skip", "--skip_frames", help="it's in the name", type=int, default=0)
    
    args = parser.parse_args()

    n_minibatches=args.number_mini_batches
    batch_size=args.batch_size
    lr=args.learning_rate
    n_classes=5
    history_length=args.history_length
    num_eval_episodes = 5
    epsilon=args.epsilon
    num_episodes= args.number_episodes
    eval_cycle = args.number_eval_cycle
    max_timesteps = args.max_timesteps
    schedule_max_timesteps = args.schedule_max_timesteps
    skip_frames = args.skip_frames
    use_pretrained = args.pretrained_weights
    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    #TODO: Define networks and load agent
    # ....
    Q_head= CNN(n_classes=n_classes, history_length=history_length)
    Q_tail = CNN(n_classes=n_classes, history_length=history_length)        
    agent = DQNAgent(Q=Q_head, Q_target=Q_tail, num_actions=n_classes, 
        batch_size=batch_size, epsilon=epsilon, 
        history_length=history_length, lr=lr)
    
    #filename = "/home/karma/Documents/DL Lab/Github/dl-lab-2022-rl/imitation_learning/models/h"+str(history_length)+"-lr0.01-agent.pt"
    filename = name = "carracing_ne"+str(num_episodes)+"_lr"+str(lr)+"_ep"+str(epsilon)+"_max_ts"+str(max_timesteps)+"_hist"+str(history_length)+"_dqn_agent.ckpt"
    agent.load(file_name=filename)

    n_test_episodes = 15
    #n_test_episodes = num_episodes

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

