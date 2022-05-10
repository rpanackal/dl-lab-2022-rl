from __future__ import print_function
from random import shuffle

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from tabulate import tabulate

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
import argparse
import torch

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    print("File reading done.")
    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def key_picker(y):
    """
    When multiple keys are pressed, one key will be kept based on priority order:
        DOWN > LEFT, RIGHT > UP
    """
    for action in y:
        if action[2]: action[:2] = 0
        elif action[0]: action[1:] = 0
    
    return y

def stack_history(X, history_length):
    X_hist = []
    for i in range(len(X)):

        if i < history_length:
            # States without enough history has first
            # state repeated
            indices = ([0]*(history_length-i-1))
            indices.extend([j for j in range(i+1)])

            X_hist.append(X[indices, ...])
        else:
            # Append past history_length images
            head = i-history_length+1
            X_hist.append(X[head:i+1, ...])
    
    return np.array(X_hist)

def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    print("...preproccessing")

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)

    # X_train = np.expand_dims(rgb2gray(X_train), 1)
    # X_valid = np.expand_dims(rgb2gray(X_valid), 1)
    
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    print("...attaching history ",history_length)
    # X_hist = []
    # for i in range(len(X_train)):

    #     if i < history_length:
    #         # States without enough history has first
    #         # state repeated
    #         indices = ([0]*(history_length-i-1))
    #         indices.extend([j for j in range(i+1)])

    #         X_hist.append(X_train[indices, ...])
    #     else:
    #         # Append past history_length images
    #         head = i-history_length+1
    #         X_hist.append(X_train[head:i+1, ...])
    
    # X_hist = np.array(X_hist) 
    X_train = stack_history(X_train, history_length)
    X_valid = stack_history(X_valid, history_length)


    #print("Samples wih history", X_hist.shape)

    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    y_train = np.apply_along_axis(action_to_id, 1, key_picker(y_train))
    y_valid = np.apply_along_axis(action_to_id, 1, key_picker(y_valid))

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    
    return X_train, y_train, X_valid, y_valid

def sample_by_weight(X, y, batch_size, n_classes):
    freq_dict = class_frequency(y)

    # Each class gets a probabilty of 1/n_classes = class_prob
    # All samples of a class divide the probability of a class : class_prob/n_samples_in_class
    class_weights = [1/(count*n_classes)for count in freq_dict["counts"]]

    p = np.take(class_weights, y)
    
    while True:
        indices = np.random.choice(np.arange(freq_dict["total"]), replace=False, 
            size=batch_size, p=p)
        yield X[indices], y[indices] 


def sample_minibatch(X, y, batch_size):

    total_n_samples = X.shape[0]
    head = 0

    permutation = np.random.permutation(total_n_samples)
    
    while True:
        if head + batch_size >= total_n_samples:
            head = 0
            permutation = np.random.permutation(total_n_samples)

        yield X[permutation[head: head+batch_size], ...], y[permutation[head: head+batch_size, ...]]

        head += batch_size 
    
def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, 
    n_classes=5, history_length=1, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(n_classes, history_length, lr)
    
    name = f"h{history_length}-bs{batch_size}-lr{lr:.4f}"

    tensorboard_eval = Evaluation(tensorboard_dir, name=name, stats=["training_loss", "training_accuracy", "validation_loss", "validation_accuracy"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser

    minibatch = sample_by_weight(X_train, y_train, batch_size, n_classes)
    minibatch_val = sample_by_weight(X_valid, y_valid, batch_size, n_classes)

    print(X_train.shape)
    #minibatch = sample_minibatch(X_train, y_train, batch_size)
    

    for i in range(n_minibatches):
        
        X_batch, y_batch = next(minibatch)

        loss, accuracy = agent.update(X_batch, y_batch)
         
        if i % 10 == 0:
            #print("Loss : ", loss.item(), " Accuracy : ", accuracy.item(), "%")
            tensorboard_eval.write_episode_data(i+1, {"training_loss": loss.item(), "training_accuracy": accuracy.item()})

            with torch.no_grad():
                X_batch_valid, y_batch_valid = next(minibatch_val)
                loss_val, accuracy_val = agent.validate(X_batch_valid, y_batch_valid)
            
            tensorboard_eval.write_episode_data(i+1, {"validation_loss": loss_val.item(), "validation_accuracy": accuracy_val.item()})


    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
      
    # TODO: save your agent
    file_name = f"h{history_length}-lr{lr}-agent.pt"
    model_dir = agent.save(os.path.join(model_dir, file_name))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-hl", "--history_length", help="it's in the name", type=int, default=1)
    parser.add_argument("-b", "--batch_size", help="it's in the name", type=int, default=128)
    parser.add_argument("-l", "--learning_rate", help="it's in the name", type=float, default=1e-2)    
    parser.add_argument("-n", "--number_mini_batches", help="it's in the name", type=int, default=1000)
    
    args = parser.parse_args()

    n_minibatches=args.number_mini_batches
    batch_size=args.batch_size
    lr=args.learning_rate
    n_classes=5
    history_length=args.history_length
    model_dir="./models"
    tensorboard_dir="./tensorboard"


    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=history_length)
    
    show_dataset_stat(y_train, y_valid)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=n_minibatches, batch_size=batch_size, lr=lr, 
        n_classes=n_classes, history_length=history_length, model_dir=model_dir, tensorboard_dir=tensorboard_dir)
    
