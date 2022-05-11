from cmath import tau
import numpy as np
import torch
import torch.optim as optim
from zmq import device
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=0, capacity=1e5, use_double=False):
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
        self.replay_buffer = ReplayBuffer(history_length, capacity=capacity)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions
        self.use_double = use_double

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.optimizer.zero_grad()
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # 2. sample next batch and perform batch update: 
        batch_states, batch_actions, batch_next_states, batch_rewards, \
            batch_dones = self.replay_buffer.next_batch(self.batch_size)
        
        batch_states = self.to_gpu(batch_states)
        batch_actions = self.to_gpu(batch_actions).long().view(batch_actions.shape[0], 1)
        #batch_actions = self.to_gpu(batch_actions)
        batch_next_states = self.to_gpu(batch_next_states)
        batch_rewards = self.to_gpu(batch_rewards)
        batch_dones = self.to_gpu(batch_dones)

        #print(batch_dones)
        #       2.1 compute td targets and loss         
        
        
        
        #import ipdb;ipdb.set_trace()
        #current_Q_values = self.predict(self.Q, batch_states)[torch.arange(len(batch_states)), batch_actions]
        current_Q_values = self.Q(batch_states).gather(1, batch_actions).squeeze(1)
        
        if not self.use_double:
            target_Q_values = self.predict(self.Q_target, batch_next_states)
            td_targets = batch_rewards + (1-batch_dones)*self.gamma*torch.max(target_Q_values, dim=1)[0]
        else:
            double_q_actions = torch.argmax(self.Q(batch_next_states), dim=1)
            target_Q_values = self.predict(self.Q_target, batch_next_states)[torch.arange(self.batch_size), double_q_actions]
            td_targets = batch_rewards + (1-batch_dones)*self.gamma*target_Q_values

        
        
        loss = self.loss_function(current_Q_values, td_targets.detach())#.float()#.detach())

        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        
        
        loss.backward()
        self.optimizer.step()
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        soft_update(self.Q_target, self.Q, tau=self.tau)
        return loss

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
            #print("Predicting Action by Network")
            state = self.to_gpu(state)
            action_id = torch.argmax(self.predict(self.Q, state)).item()
        else:
            #pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            #action_id = np.random.randint(self.num_actions)
            #print("Number of actions: ", self.num_actions)
            action_id = np.random.randint(self.num_actions)
            #print("Random action")
            #print("Coming from else: ", action_id)

        return action_id

    def predict(self, net, state):
        #x = torch.from_numpy(x).to(self.device).float()        
        if len(state.shape)==3:
            state=state[np.newaxis, ...]
        #if self.num_actions>2:
        #    print(state.shape)
            #state = torch.moveaxis(state, 3, 1)
        return net(state)
    
    def to_gpu(self, x):
        return torch.tensor(x).to(self.device).float()
    
    def to_cpu(self, x):
        return x.numpy()

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))

