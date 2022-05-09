import numpy as np
from tabulate import tabulate

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif np.allclose(a, [0.0, 0.0, 0.2], 1e-8): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])
    

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))

def class_frequency(y, n_classes=5):
    """
    A getter function for class-wise frequencies from 1-D parameter y
    with scalar class ids.

    Returns:
        A dictionary with keys:
            "counts" - type: list, description: frequencies of classes indexed by 
                scalar class ids.
            "total"  - type: int, description: length of parameter y
    """

    counts = [0]*n_classes
    total = len(y)
    for i in range(total): counts[y[i]] += 1

    stats = {
        "counts": counts,
        "total": total
    }
    return stats

def show_dataset_stat(y_train, y_valid):
    """
    Show class-wise frequency stats of training and validation dataset.
    """
    classes = ["STRAIGHT", "LEFT", "RIGHT",  "ACCELERATE", "BREAK"]

    train_stats = class_frequency(y_train)
    valid_stats = class_frequency(y_valid)

    table = list(zip(classes, train_stats["counts"], valid_stats["counts"]))
    table.append(["Total", train_stats["total"], valid_stats["total"]])

    header = ["Class", "Training Set", "Validation Set"]
    
    print("\nClass-wise frequency table :\n", tabulate(table, headers=header, tablefmt="pretty"))

