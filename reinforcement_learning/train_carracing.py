# export DISPLAY=:0 

import sys
from unittest import skip

sys.path.append("../") 

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
#import torch

import argparse

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    #state = torch.moveaxis(torch.tensor(state), 3, 1)
    
    while True:
        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state, deterministic)        
        action = better_id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, num_eval_episodes=20, skip_frames=0,
                eval_cycle=5, model_dir="./models_carracing", tensorboard_dir="./tensorboard",
                epsilon_max=0.1, lr=0.0001, max_timesteps=1000, schedule_max_ts=False):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    name = "carracing_ne"+str(num_episodes)+"_lr"+str(lr)+"_ep"+str(epsilon_max)+"_max_ts"+str(max_timesteps)+"_hist"+str(history_length)
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), name, ["episode_reward", "straight", "left", "right", "accel", "brake", "lr", "epsilon", "history_length", "scheduling_max_timesteps", "skip_frames"])
    tensorboard_val = Evaluation(os.path.join(tensorboard_dir, "val"), name, ["episode_reward", "straight", "left", "right", "accel", "brake", "lr", "epsilon", "history_length", "scheduling_max_timesteps", "skip_frames"])

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        """
        linearly increaing max_timesteps
        """
        if schedule_max_ts:
            b = (i+1)/num_episodes
            max_timesteps = (1-((1-b)*0.9 + b*0.001))*1000        

        stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, rendering=True, do_training=True, skip_frames=skip_frames)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE),
                                                      "lr": lr, 
                                                      "epsilon": epsilon, 
                                                      "history_length": history_length,
                                                      "scheduling_max_timesteps": schedule_max_ts,
                                                      "skip_frames": skip_frames
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats_val = run_episode(env, agent, deterministic=True, do_training=False)
                tensorboard_val.write_episode_data(i, eval_dict={ "episode_reward" : stats_val.episode_reward, 
                                                      "straight" : stats_val.get_action_usage(STRAIGHT),
                                                      "left" : stats_val.get_action_usage(LEFT),
                                                      "right" : stats_val.get_action_usage(RIGHT),
                                                      "accel" : stats_val.get_action_usage(ACCELERATE),
                                                      "brake" : stats_val.get_action_usage(BRAKE),
                                                      "lr": lr, 
                                                      "epsilon": epsilon, 
                                                      "history_length": history_length,
                                                      "scheduling_max_timesteps": schedule_max_ts,
                                                      "skip_frames": skip_frames
                                                      })

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            model_name = name + "_dqn_agent.ckpt"
            agent.save(agent.sess, os.path.join(model_dir, model_name)) 
        """
        Schedule for epsilon
        agent.epsilon
        """

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-hl", "--history_length", help="it's in the name", type=int, default=1)
    parser.add_argument("-b", "--batch_size", help="it's in the name", type=int, default=128)
    parser.add_argument("-l", "--learning_rate", help="it's in the name", type=float, default=1e-2)    
    parser.add_argument("-n", "--number_mini_batches", help="it's in the name", type=int, default=1000)
    parser.add_argument("-ec", "--number_eval_cycle", help="it's in the name", type=int, default=10)
    parser.add_argument("-e", "--epsilon", help="decide greedy approach threshold", type=float, default=0.1)
    parser.add_argument("-ne", "--number_episodes", help="it's in the name", type=int, default=10)
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

    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q_head= CNN(n_classes=n_classes, history_length=history_length)
    Q_tail = CNN(n_classes=n_classes, history_length=history_length)        
    agent = DQNAgent(Q=Q_head, Q_target=Q_tail, num_actions=n_classes, 
        batch_size=batch_size, epsilon=epsilon, 
        history_length=history_length, lr=lr)
    if use_pretrained and (history_length==1 or history_length==3 or history_length==5):        
        filename = "/home/karma/Documents/DL Lab/Github/dl-lab-2022-rl/imitation_learning/models/h"+str(history_length)+"-lr0.01-agent.pt"
        agent.load(file_name=filename)
    
    train_online(env, agent, num_episodes=num_episodes, 
                history_length=history_length, eval_cycle=eval_cycle, 
                num_eval_episodes=num_eval_episodes, skip_frames = skip_frames,
                model_dir="./models_carracing", epsilon_max=epsilon, lr=lr, 
                max_timesteps=max_timesteps, schedule_max_ts=schedule_max_timesteps)

