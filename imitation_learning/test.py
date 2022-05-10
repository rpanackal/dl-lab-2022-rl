from __future__ import print_function
from asyncio.log import logger

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch

from tensorboard_evaluation import Evaluation
from agent.bc_agent import BCAgent
from utils import *
import argparse


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=1):
    
    history = []

    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    # forced_accel = True

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...
        #state = np.expand_dims(rgb2gray(state), (0,1))
        
        state = rgb2gray(state)
        
        if history:
            history.append(state)
            history.pop(0)
        else:
            history = [state for i in range(history_length)]

        #print("length of history", len(history))
        state = np.array(history)[np.newaxis, ...]
        #print("transformed state",state.shape)
        #print("After expanding dimension- Shape", state.shape, " Type:", type(state))
        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        # if not forced_accel:
        output = agent.predict(state)
        prediction = torch.argmax(output)
        a = id_to_action(prediction)
        # else:
        #     a = np.array([0.0, 1, 0.0])
        #     forced_accel = False

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps:
            # forced_accel = True
            break

    return episode_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-hl", "--history_length", help="it's in the name", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", help="it's in the name", type=float, default=1e-2)    
    args = parser.parse_args()
    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    history_length = args.history_length
    lr = args.learning_rate
    model_dir="./models"
    tensorboard_dir="./tensorboard"
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent(history_length=history_length)
    name = f"test-h{history_length}-agent"
    tensorboard_eval = Evaluation(tensorboard_dir, name=name, stats=["episode_id", "episode_return", "history_length"])
    model_name=f"{model_dir}/h{history_length}-lr{lr}-agent.pt"
    agent.load(model_name)

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, history_length=history_length)
        tensorboard_eval.write_episode_data(i+1, {"episode_id": i, "episode_return": episode_reward, "history_length": history_length})
        episode_rewards.append(episode_reward)
        

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    results["history length"]=history_length
 
    fname = "./results/history_%d_results_bc_agent-%s.json" %(history_length, datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
