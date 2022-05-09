from __future__ import print_function

import argparse
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import time


def key_press(k, mod):
    global restart, end

    if k == k==key.ESCAPE: end = True
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.2

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0.0
    if k == key.UP:    a[1] = 0.0
    if k == key.DOWN:  a[2] = 0.0


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


if __name__ == "__main__":
    restart = False
    end = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = gym.make('CarRacing-v0').unwrapped

    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release


    a = np.array([0.0, 0.0, 0.0]).astype('float32')
    
    episode_rewards = []
    steps = 0

    while not end:

        episode_reward = 0
        state = env.reset()

        #Track starting index of an episode's data in samples dictionary
        episode_head = steps 
        print("episode head at ", episode_head, "and #returns is ",
            len(episode_rewards))
        
        while not end:
            
            time.sleep(0.025) # to control the latency of key press

            next_state, r, done, info = env.step(a)
            episode_reward += r

            samples["state"].append(state)            # state has shape (96, 96, 3)
            samples["action"].append(np.array(a))     # action has shape (1, 3)
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            env.render()

            if steps % 1000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if restart:
                restart = False
                for label in samples.keys(): 
                    samples[label] = samples[label][:episode_head]
                
                steps = episode_head
                break

            # At episode termination store episode return and data.
            if done:
                episode_rewards.append(episode_reward)

                if args.collect_data:
                    print('... saving data after ',steps,'steps.')
                    store_data(samples, "./data")
                    save_results(episode_rewards, "./results")

                break

    env.close()

    

   
