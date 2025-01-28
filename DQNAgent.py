import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import csv
from Environment import SCIONEnvironment

# Set up the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.002
gamma = 0.8
epsilon = 0.995
epsilon_min = 0.1
epsilon_max = 0.995
epsilon_increment = 0.30
epsilon_decay = 0.925
batch_size = 32
replay_memory_buffer = 100000
target_update_frequency = 2
previous_reward = 0
epsilon_decreased = 0
explorations = 0
exploitations = 0
total_actions = 0

# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)  # Input → Hidden
        self.fc2 = nn.Linear(64, 128)         # Hidden → Hidden
        self.fc3 = nn.Linear(128, 64)        # Hidden → Hidden
        self.fc4 = nn.Linear(64, output_shape)  # Hidden → Output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # Linear activation for output

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_buffer = deque(maxlen=replay_memory_buffer)
        self.recent_rewards = deque(maxlen=3)  # Track last 3 rewards
        self.priority_buffer = deque(maxlen=replay_memory_buffer)  # Priority scores
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.previous_reward = previous_reward
        self.epsilon_decay = epsilon_decay
        self.path_counts = np.zeros(action_size)  # Track path selection counts
        self.epsilon_priority = 1e-5  # Small value to avoid zero probabilities
        self.alpha = 0.6  # Controls how much prioritization is used (0 = uniform sampling)

        self.model = QNetwork(state_size, action_size).to(device)
        print(f"Model is on GPU: {next(self.model.parameters()).is_cuda}")
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_model()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))
        # Assign initial priority
        if len(self.priority_buffer) > 0:
            max_priority = max(self.priority_buffer)
        else:
            max_priority = 1.0  # Default high priority for new experiences
        self.priority_buffer.append(max_priority)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            chosen_action = random.randrange(self.action_size)
            print(f'\nChosen action (exploration):{chosen_action}' )
            flag = True
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                act_values = self.model(state_tensor).cpu().numpy()
            act_values += self.path_counts * 1e-5  # Add small bias for path counts to break ties
            chosen_action = np.argmax(act_values)
            flag = False

            print(f'\nChosen action (exploitation):{chosen_action}' )
            print(f'Model output values: {act_values}')

        self.path_counts[chosen_action] += 1  # Update path count for chosen action

    
        return chosen_action, flag

    def replay(self, batch_size):
        if len(self.memory_buffer) < batch_size:
            return

        # Convert priorities to probabilities
        priorities = np.array(list(self.priority_buffer), dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()  # Normalize to sum to 1

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory_buffer), size=batch_size, p=probabilities)

        # Gather minibatch
        minibatch = [self.memory_buffer[idx] for idx in indices]
        importance_sampling_weights = (1 / (len(self.memory_buffer) * probabilities[indices])) ** (1 - self.alpha)
        importance_sampling_weights /= importance_sampling_weights.max()  # Normalize weights

        # Process the minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(device).squeeze(1)
        actions = torch.LongTensor(actions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device).squeeze(1)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q-values
        q_values = self.model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        current_q_values = q_values.gather(1, actions)
        target_q_values = target_q_values.unsqueeze(1)

        # Compute loss and apply importance sampling weights
        loss = self.loss_fn(current_q_values, target_q_values)
        loss = (loss * torch.FloatTensor(importance_sampling_weights).to(device)).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy().flatten()  # Flatten to ensure 1D array
        for i, idx in enumerate(indices):
            self.priority_buffer[idx] = float(td_errors[i]) + self.epsilon_priority  # Ensure scalar values


        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/"
    )
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    episodes = 1000

    with open("episode_stats.csv", "w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(["Episode", "Total_Actions", "Explorations", "Exploitations", "Actions_Taken"])

        for e in range(episodes):
            agent.epsilon = 0.995 
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            # Reset tracking variables
            total_actions, explorations, exploitations = 0, 0, 0
            actions_taken = []  # Track the sequence of actions

            for time in range(800):
                print(state)

                action, action_type = agent.act(state)
                actions_taken.append((action,action_type,state[0][0],state[0][1]))  # Append chosen action
                next_state, reward, done = env.step(action)

                # Add reward to recent rewards and calculate rolling average
                agent.recent_rewards.append(reward)
                rolling_avg_reward = np.mean(agent.recent_rewards)

                # Adjust epsilon based on reward compared to rolling average
                if reward < rolling_avg_reward - 0.05 and action_type is False:
                    print("Reward decreased significantly, increasing exploration")
                    epsilon_decreased += 1
                    agent.epsilon = min(agent.epsilon + agent.epsilon_increment, agent.epsilon_max)
                elif reward >= rolling_avg_reward:
                    if agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
                if action_type is False:
                    exploitations += 1
                else:
                    explorations += 1
                agent.previous_reward = reward  # Update previous reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                ##Delete this update after having traffic simulation on the network
                ##Then we dont need to update the paths parameters
                env.paths = env.get_paths()
                if env.simulation_complited:
                    agent.save(f"dqn_model_{e}.pth")
                    print("Simulation completed")
                    total_actions = len(actions_taken)
                    print(f'Epsilon decreasde {epsilon_decreased} times')
                    writer.writerow([e, total_actions, explorations, exploitations, actions_taken])

                    exit()

                if len(agent.memory_buffer) > batch_size:
                    agent.replay(batch_size)
                    if batch_size < 512:
                        batch_size += 1


            # Track episode stats
            total_actions = len(actions_taken)
            writer.writerow([e, total_actions, explorations, exploitations, actions_taken])

            if e % target_update_frequency == 0:
                agent.update_target_model()

            if e % 1 == 0:
                agent.save(f"dqn_model_{e}.pth")

