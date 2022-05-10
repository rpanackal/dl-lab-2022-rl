from cmath import tau
import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length, capacity=1e5)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # 2. sample next batch and perform batch update: 
        batch_states, batch_actions, batch_next_states, batch_rewards, \
            batch_dones = self.replay_buffer.next_batch(self.batch_size)
        #       2.1 compute td targets and loss         
        current_Q_values = self.Q(batch_states)[torch.arange(self.batch_size), batch_actions]

        target_Q_values = self.Q_target(batch_next_states) 
        #import ipdb;ipdb.set_trace()
        td_targets = batch_rewards + (1-batch_dones)*self.gamma*torch.max(target_Q_values).item()#.detach().cpu(), dim=1)
        
        loss = self.loss_function(current_Q_values, torch.tensor(td_targets, device=self.device)).float()#.detach())

        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        self.optimizer.zero_grad()
        loss=loss.float()
        loss.backward()
        self.optimizer.step()
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        soft_update(self.Q_target, self.Q, tau=self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            #pass
            # TODO: take greedy action (argmax)
            
            action_id = torch.argmax(self.predict(self.Q, state))
        else:
            #pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            #action_id = np.random.randint(self.num_actions)
            #print("Number of actions: ", self.num_actions)
            action_id = torch.randint(low=0, high=self.num_actions+1, size=[], device=self.device)
            #print("Coming from else: ", action_id)

        return action_id

    def predict(self, net, x):
        x = torch.from_numpy(x).to(self.device).float()
        return net(x)

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
