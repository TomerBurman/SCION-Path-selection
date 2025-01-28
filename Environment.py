import requests
import numpy as np
import logging
from traffic_model import run_simulation, get_current_simulated_time  # Import the generator and time function
NUMBER_OF_DAYS_IN_WEEK = 6
HOURS_OF_DAY_REFACTORED = 47

class SCIONEnvironment:
    def __init__(self, sender_url, receiver_url, paths_url, path_selection_url):
        self.sender_url = sender_url
        self.receiver_url = receiver_url
        self.paths_url = paths_url
        self.path_selection_url = path_selection_url
        self.state_size = 5
        self.paths = self.get_paths()
        self.action_size = len(self.paths)
        self.last_action = None
        self.steps_counter = 0
        self.simulation_complited =False

        # Initialize the traffic model generator
        self.traffic_model = run_simulation()
        self.current_time = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_paths(self):
        try:
            response = requests.get(self.paths_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch paths: {e}")
            return {}

    def reset(self):
        # Advance the traffic model to synchronize with the simulation
        self.advance_traffic_model()
        return self.get_state()

    def advance_traffic_model(self):
        try:
            self.current_time = next(self.traffic_model)
        except StopIteration:
            self.logger.info("Traffic model simulation completed.")
            self.simulation_complited =True
            self.current_time = get_current_simulated_time()  # Get the last simulated time

    def get_state(self, response=None, path_index=None):
        if response is None:
            return np.array([0, 0, 0, 0, 0])

        self.logger.info(f'Stats: {response}')

        # Advance the traffic model and get the current simulated time
        self.steps_counter += 1
        if self.steps_counter % 1 == 0:
            self.advance_traffic_model()
            self.steps_counter = 0
        
        day_of_week = self.current_time.weekday() / NUMBER_OF_DAYS_IN_WEEK
        current_hour = self.current_time.hour
        current_minute = self.current_time.minute
        time_of_day = self.get_time_of_day(current_hour, current_minute) / HOURS_OF_DAY_REFACTORED

        goodput = response.get('goodput_received_mbps', 0)
        self.logger.info(f"Goodput: {goodput}")
        goodput_normalized = self.normalize_value(goodput, max_value=50)
        goodput_range = self.get_range(goodput_normalized, max_value=1)

        path = self.paths.get(f'{path_index}', self.paths.get('0', {}))
        capacity = path.get('bandwidth_mbps', 0)
        capacity_normalized = self.normalize_value(capacity, max_value=50)
        capacity_range = self.get_range(capacity_normalized, max_value=1)

        average_delay = response.get("average_delay", 0)
        loss_percent = response.get('packet_loss', 0)
        link_trust = self.calculate_link_trust(loss_percent, average_delay)
        link_trust_normalized = self.normalize_value(link_trust, max_value=1)
        link_trust_range = self.get_range(link_trust_normalized, max_value=1)

        state = np.array([day_of_week, time_of_day, goodput_range, capacity_range, link_trust_range])
        self.logger.info(f"day of the week: {day_of_week}")
        self.logger.info(f"State: {state}")
        return state

    def get_range(self, value, max_value, num_ranges=40):
        step = max_value / num_ranges
        for i in range(1, num_ranges + 1):
            if value <= i * step:
                return (i - 1) / num_ranges
        return (num_ranges - 1) / num_ranges

    def get_time_of_day(self, current_hour, current_minute):
        # returns time state in the range of 0-48
        return current_hour *2 +(current_minute//30) 


    def calculate_link_trust(self, packet_loss, average_delay, packet_loss_weight=0.5, delay_weight=0.5):
        if average_delay is None:
            average_delay = 0
            self.logger.error(f"average_delay is None")

        link_trust = 1 - ((packet_loss_weight * packet_loss) + (delay_weight * average_delay))
        return max(0, min(link_trust, 1))

    def normalize_value(self, value, max_value):
        return min(max(value / max_value, 0), 1)

    def step(self, action, duration=0.1):
        path = self.paths.get(f'{action}', {})
        try:
            if action != self.last_action:
                requests.get(self.path_selection_url + f'{action}')
                self.last_action = action  # Update last action

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to select path: {e}")

        try:
            response = requests.post(
                f"{self.sender_url}/send",
                json={"rate": path.get('bandwidth_mbps', 0), "duration": duration}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to send data: {e}")
            response = {}

        new_state = self.get_state(response.json(), path_index=action)
        reward = self.calculate_reward(new_state)
        done = self.is_done(new_state)

        return new_state, reward, done

    def calculate_reward(self, state, goodput_weight=0.7, link_trust_weight=0.3):
        goodput = state[2]
        link_trust = state[4]

        reward = 2 * ((goodput * goodput_weight) + (link_trust * link_trust_weight)) - 1
        self.logger.info(f"Reward: {reward}")
        return reward

    def is_done(self, state):
        return False

    def render(self):
        self.logger.info(f"Current State: {self.get_state()}")
