import os
from Environment import SCIONEnvironment
from Dijkstra_routing import DijkstraRouter
from DQNAgent import DQNAgent
import torch
import csv
import requests
import random
import numpy as np


# Set up the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_data_and_record(env, action, data_size, max_rate):
    """
    Sends data using POST request and records the results.
    Args:
        env (SCIONEnvironment): The simulation environment.
        action (int): The selected action/path.
        data_size (int): The size of data to send.
        max_rate (int): The maximum data rate.
    Returns:
        dict: A dictionary containing the results of the data transmission.
    """
    try:
        response = requests.post(
            f"{env.sender_url}/send",
            json={"duration": 1.0, "size": data_size}
        )
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")
        response_data = {}

    return {
        "timestamp": env.current_time,
        "path": action,
        "data_size": data_size,
        "average_delay": response_data.get("average_delay", None),
        "elapsed_time": response_data.get("elapsed_time", None),
        "goodput_received_mbps": response_data.get("goodput_received_mbps", None),
        "goodput_sent_mbps": response_data.get("goodput_sent_mbps", None),
        "packet_loss": response_data.get("packet_loss", None),
        "total_bytes_received": response_data.get("total_bytes_received", None),
        "total_bytes_sent": response_data.get("total_bytes_sent", None)
    }

def calculate_reward(path):
    """
    Calculate reward using 70% bandwidth and 30% link trust.
    Args:
        path (dict): Path information containing bandwidth and latency data.
    Returns:
        float: Calculated reward.
    """
    bandwidth = path.get("bandwidth_mbps", 0)
    link_trust = 1 - (
        0.5 * path.get("loss_percent", 0) +
        0.5 * path.get("latency_ms", 0) / ((len(path.get("hops"))/2) * 500)
    )
    bandwidth_range = env.get_range(bandwidth / 50, 1)
    link_trust_range = env.get_range(link_trust, 1)
    link_trust = max(0, min(link_trust, 1))  # Ensure link trust is between 0 and 1
    return 0.7 * bandwidth_range + 0.3 * link_trust_range

def find_top_3_paths(env):
    """
    Determine the top 3 paths according to the custom reward function.
    Args:
        env (SCIONEnvironment): The simulation environment.
    Returns:
        list: A list of the top 3 path IDs.
    """
    paths = env.get_paths()
    sorted_paths = sorted(
        paths.items(),
        key=lambda x: calculate_reward(x[1]),
        reverse=True
    )

    return [path[0] for path in sorted_paths]


def send_path_change_request(path_id,env):
        """
        Directly send a request to the environment to change the selected path.
        """
        try:
            response = requests.get(f"{env.path_selection_url}{path_id}")
            response.raise_for_status()
            print(f"Path changed successfully to path ID: {path_id}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to change path: {e}")

def evaluate_model(model_type, env, output_file):
    """
    Evaluate the performance of the model and save the results.
    Args:
        model_type (str): Either 'dqn' or 'dijkstra'.
        env (SCIONEnvironment): The simulation environment.
        output_file (str): The output file name to save results.
    """
    # advancing the traffic model for initializing the environment current_time 
    env.advance_traffic_model()

    if model_type == "dqn":
        # Load the pre-trained DQN model
        model_path = "./dqn_model_16.pth"
        agent = DQNAgent(env.state_size, env.action_size)
        agent.load(model_path)

        def select_action(state):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                act_values = agent.model(state_tensor).cpu().numpy()

            chosen_action = np.argmax(act_values)
            send_path_change_request(path_id=f'{chosen_action}',env = env )

            return str(chosen_action)

    

    elif model_type == "dijkstra":
        # Initialize Dijkstra routing
        router = DijkstraRouter(env, metric="latency")

        def select_action(state):
            best_path, best_path_id, best_cost = router.find_shortest_path()
            if (
                best_cost < router.current_best_cost
                or best_path_id != router.current_best_path_id
            ):
                router.logger.info(
                    f"Path updated! New best path based on {router.metric}: {best_path_id, best_path} with cost: {best_cost}"
                )
                router.current_best_path_id = best_path_id
                router.current_best_cost = best_cost
                router.send_path_change_request(best_path_id)
            return best_path_id
        

    elif model_type == "random":
        def select_action(state):
            chosen_action = random.randrange(env.action_size)
            send_path_change_request(path_id=f'{chosen_action}',env = env)
            return str(chosen_action)
    
    elif model_type == "optimal":
        def select_action(state):
            chosen_action = find_top_3_paths(env)[0]
            send_path_change_request(path_id=f'{chosen_action}',env = env)
            return str(chosen_action)

    else:
        raise ValueError("Invalid model type. Choose 'dqn' or 'dijkstra'.")

    results = []
    traffic_model_done = False
    max_rate = 50
    data_sizes = [10, 20, 30, 50]

    total_decisions = 0
    top_1_selection_count = 0
    top_2_selection_count = 0
    top_3_selection_count = 0
    top_4_selection_count = 0
    top_5_selection_count = 0
    top_6_selection_count = 0
    

    while not traffic_model_done:
        for data_size in data_sizes:
            
            try:
                response = requests.post(
                    f"{env.sender_url}/send",
                    json={"rate": max_rate, "duration": 0.5}
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                env.logger.error(f"Failed to send data: {e}")
                response = {}

            state = env.get_state(response = response.json())
            action = select_action(state)
            result = send_data_and_record(env, action, data_size, max_rate)
            results.append(result)

            # Find top 3 paths
            top_3_paths = find_top_3_paths(env)
            total_decisions += 1

            if action in top_3_paths:
                if action == top_3_paths[0]:
                    top_1_selection_count += 1
                elif action == top_3_paths[1]:
                    top_2_selection_count += 1
                elif action == top_3_paths[2]:
                    top_3_selection_count += 1
                elif action == top_3_paths[3]:
                    top_4_selection_count += 1
                elif action == top_3_paths[4]:
                    top_5_selection_count += 1
                elif action == top_3_paths[5]:
                    top_6_selection_count += 1
                

            print(f"\nData sent information\nData Size: {data_size} | Path: {action} | Time: {env.current_time}")
            print(f"Top 3 paths: {top_3_paths}")

            # Check if traffic simulation has ended
            if env.simulation_complited:
                traffic_model_done = True
                break

    # Calculate selection percentages
    top_1_percentage = (top_1_selection_count / total_decisions) * 100 if total_decisions > 0 else 0
    top_2_percentage = (top_2_selection_count / total_decisions) * 100 if total_decisions > 0 else 0
    top_3_percentage = (top_3_selection_count / total_decisions) * 100 if total_decisions > 0 else 0

    print(f"Top 1 selection percentage: {top_1_percentage:.2f}%")
    print(f"Top 2 selection percentage: {top_2_percentage:.2f}%")
    print(f"Top 3 selection percentage: {top_3_percentage:.2f}%")

    # Save results to a CSV file
    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Save top path percentages to a summary file
    with open(output_file.replace(".csv", "_summary.txt"), "w") as summary_file:
        summary_file.write(f"Top 1 selection percentage: {top_1_percentage:.2f}%\n")
        summary_file.write(f"Top 2 selection percentage: {top_2_percentage:.2f}%\n")
        summary_file.write(f"Top 3 selection percentage: {top_3_percentage:.2f}%\n")

    print(f"Simulation with {model_type} completed. Results saved to {output_file}.")

if __name__ == "__main__":
    # Initialize the environment
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/"
    )

    # Evaluate the DQN model
    dqn_output_file = "dqn_simulation_results.csv"
    evaluate_model("dqn", env, dqn_output_file)

    # # Evaluate the Dijkstra routing
    # dijkstra_output_file = "dijkstra_simulation_results.csv"
    # evaluate_model("dijkstra", env, dijkstra_output_file)

    # # Evaluate the random routing
    # random_output_file = "random_simulation_results.csv"
    # evaluate_model("random", env, random_output_file)

    # # Evaluate the optimal routing
    # optimal_output_file = "optimal_simulation_results.csv"
    # evaluate_model("optimal", env, optimal_output_file)