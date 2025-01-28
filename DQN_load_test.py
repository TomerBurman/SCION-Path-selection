import datetime
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from Environment import SCIONEnvironment
from simulate_data_sending import DataSender

# Set up the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.0005
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 0.995
epsilon_increment = 0.05
epsilon_decay = 0.995
batch_size = 64
replay_memory_buffer = 100000
target_update_frequency = 1
previous_reward = 0



# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_buffer = deque(maxlen=replay_memory_buffer)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.previous_reward = previous_reward
        self.epsilon_decay = epsilon_decay
        self.path_counts = np.zeros(action_size)  # Track path selection counts

        self.model = QNetwork(state_size, action_size).to(device)
        print(f"Model is on GPU: {next(self.model.parameters()).is_cuda}")
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def remember(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))


    def act(self, state):
      
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            act_values = self.model(state_tensor).cpu().numpy()

       
        chosen_action = np.argmax(act_values)


        print(f'\nChosen action (exploitation):{chosen_action}' )
        print(f'Model output values: {act_values}')

        self.path_counts[chosen_action] += 1  # Update path count for chosen action

    
        return chosen_action

    def replay(self, batch_size):
        if len(self.memory_buffer) < batch_size:
            return

        minibatch = random.sample(self.memory_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(device).squeeze(1)
        actions = torch.LongTensor(actions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device).squeeze(1)
        dones = torch.FloatTensor(dones).to(device)

        q_values = self.model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        current_q_values = q_values.gather(1, actions)
        target_q_values = target_q_values.unsqueeze(1)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":

    # Helper function to calculate link trust
    def calculate_link_trust(packet_loss, average_delay, packet_loss_weight=0.5, delay_weight=0.5):
        link_trust = 1 - ((packet_loss_weight * packet_loss) + (delay_weight * average_delay))
        return max(0, min(link_trust, 1))
    
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/"
    )
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    agent.load('dqn_model_15.pth')
    episodes = 1000

    total_decisions = 0
    top_1_selection_count = 0
    top_2_selection_count = 0
    top_3_selection_count = 0
    top_1_percentage = 0
    top_2_percentage = 0
    top_3_percentage = 0

    simulation_ended = False

    # Run the data sending simulation
    data_sender = DataSender(
    env=env,
    interval_minutes=1,
    data_sizes=[500, 1000, 1500],
    duration=1.0,
    output_file="dqn_results.csv"
    )
    send_data_thread = threading.Thread(target=data_sender.send_data, daemon=True)
    send_data_thread.start()

    for e in range(episodes):
        if simulation_ended:
            break
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            ##Delete this update after having traffic simulation on the network
            ##Then we dont need to update the paths parameters
            env.paths = env.get_paths()



            # Sort paths by reward function (70% bandwidth, 30% link trust)
            sorted_paths = sorted(
                env.paths.items(),
                key=lambda x: 0.7 * x[1].get('bandwidth_mbps', 0) + 0.3 * calculate_link_trust(
                    x[1].get('loss_percent', 0),
                    x[1].get('latency_ms', 0)
                ),
                reverse=True
            )[:3]

            # Extract path IDs for the top 3
            top_3_paths = [int(path[0]) for path in sorted_paths]
            top_1_path = top_3_paths[0] if top_3_paths else None
            top_2_path = top_3_paths[1] if top_3_paths else None
            top_3_path = top_3_paths[2] if top_3_paths else None

            # Extract path IDs and calculated rewards for the top 3
            top_paths_info = [(int(path[0]),path[1].get('bandwidth_mbps', 0),
                calculate_link_trust(path[1].get('loss_percent', 0),2 * path[1].get('latency_ms', 0)),
                0.7 * path[1].get('bandwidth_mbps', 0) + 0.3 * calculate_link_trust(
                    path[1].get('loss_percent', 0),
                    path[1].get('latency_ms', 0)
                )-1)
                for path in sorted_paths
            ]

            # Update counters
            total_decisions += 1
            if action == top_1_path:
                top_1_selection_count += 1
            elif action  == top_2_path:
                top_2_selection_count += 1
            elif action == top_3_path:
                top_3_selection_count += 1

            if time % 5 == 0:
                top_1_percentage = (top_1_selection_count / total_decisions) * 100 if total_decisions > 0 else 0
                top_2_percentage = (top_2_selection_count / total_decisions) * 100 if total_decisions > 0 else 0
                top_3_percentage = (top_3_selection_count / total_decisions) * 100 if total_decisions > 0 else 0
                print(f"\n### Episode {e}: Top 1 selection percentage: {top_1_percentage:.2f}%")
                print(f"### Episode {e}: Top 2 selection percentage: {top_2_percentage:.2f}%")
                print(f"### Episode {e}: Top 3 selection percentage: {top_3_percentage:.2f}%")

            # Print the top 3 paths, their bandwidth, link trust, and reward
            print(f"Top 3 paths by reward function (70% bandwidth, 30% link trust):")
            for path_id, bandwidth, link_trust, reward in top_paths_info:
                print(f"Path {path_id}: Bandwidth = {bandwidth:.2f} Mbps, Link Trust = {link_trust:.4f}, Reward = {reward:.4f}")

        if env.simulation_complited:
            # Get the current date and time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Generate the filename with the timestamp
            filename = f"percentages_{timestamp}.txt"

            with open(filename, "w") as file:
                file.write(f"Top 1 selection percentage: {top_1_percentage:.2f}%\n")
                file.write(f"Top 2 selection percentage: {top_2_percentage:.2f}%\n")
                file.write(f"Top 3 selection percentage: {top_3_percentage:.2f}%\n")
            print(f"Simulation ended. Percentages saved in {filename}.")
            simulation_ended = True
            break

    send_data_thread.join()  # Ensure data transmission thread completes