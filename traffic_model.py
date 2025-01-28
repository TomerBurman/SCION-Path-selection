import requests
import datetime
import time
import random
import numpy as np
from tqdm import tqdm

# Configuration
API_URL = 'http://10.101.0.71:8050/set_link'

# Link Definitions
LINKS = [
    {'id': 'ix200', 'category': 'edge'},
    {'id': 'ix201', 'category': 'edge'},
    {'id': 'ix202', 'category': 'edge'},
    {'id': 'ix205', 'category': 'edge'},
    {'id': 'ix206', 'category': 'core'},
    {'id': 'ix207', 'category': 'core'},
    {'id': 'ix208', 'category': 'core'},
    {'id': 'ix209', 'category': 'core'},
    {'id': 'ix203', 'category': 'access'},
    {'id': 'ix204', 'category': 'access'},
]

# Simulation Parameters
SIMULATION_DURATION = 7 * 24 * 60 * 60   # Simulate 7 days (in seconds)
TIME_SCALE = 5 * 60  # 1 real second = 15 * 60 simulated seconds
TIME_STEP = 15 * 60  # Simulate in 15-minute increments (in simulated seconds)

# Random Seed for Repeatability
SEED = 42  # You can change this seed to get different repeatable runs

# Set seeds for random number generators
random.seed(SEED)
np.random.seed(SEED)

# Start time in simulated time
SIMULATED_TIME_START = datetime.datetime(2023, 10, 1, 0, 0, 0)  # Arbitrary start date
current_simulated_time = SIMULATED_TIME_START

# Function to get current simulated time and day
def get_current_simulated_time():
    return current_simulated_time

# Link Category Configurations
LINK_CATEGORIES = {
    'core': {
        'min_bw': 20, 'max_bw': 50,  # in Mbps
        'min_latency': 5, 'max_latency': 200,  # in ms
        'min_jitter': 0, 'max_jitter': 0,  # in ms
        'min_loss': 0.0, 'max_loss': 0.1,  # in %
    },
    'edge': {
        'min_bw': 10, 'max_bw': 50,
        'min_latency': 5, 'max_latency': 500,
        'min_jitter': 0, 'max_jitter': 0,
        'min_loss': 0.0, 'max_loss': 1,
    },
    'access': {
        'min_bw': 7, 'max_bw': 35,
        'min_latency': 5, 'max_latency': 100,
        'min_jitter': 0, 'max_jitter': 0,
        'min_loss': 0.0, 'max_loss': 5,
    },
}

# Markov Model States
STATES = ['Normal', 'Congested', 'Overloaded']
STATE_TRANSITIONS = {
    'Normal': {'Normal': 0.85, 'Congested': 0.10, 'Overloaded': 0.05},
    'Congested': {'Normal': 0.10, 'Congested': 0.80, 'Overloaded': 0.10},
    'Overloaded': {'Normal': 0.05, 'Congested': 0.15, 'Overloaded': 0.80},
}

# Initialize link states
link_states = {link['id']: random.choice(STATES) for link in LINKS}

# Function to update link state using Markov chain
def update_link_state(link_id, is_peak):
    current_state = link_states[link_id]
    transitions = STATE_TRANSITIONS[current_state].copy()

    # Adjust transition probabilities during peak times
    if is_peak:
        if current_state == 'Normal':
            transitions = {'Normal': 0.80, 'Congested': 0.15, 'Overloaded': 0.05}
        elif current_state == 'Congested':
            transitions = {'Normal': 0.05, 'Congested': 0.85, 'Overloaded': 0.10}
        elif current_state == 'Overloaded':
            transitions = {'Normal': 0.05, 'Congested': 0.10, 'Overloaded': 0.85}

    next_state = random.choices(
        population=list(transitions.keys()),
        weights=list(transitions.values())
    )[0]
    link_states[link_id] = next_state
    return next_state

# Function to check if current time is peak for a link
def is_peak_time(link_category, sim_time):
    weekday = sim_time.weekday()
    hour = sim_time.hour
    # Access Links
    if link_category == 'access':
        if weekday >= 5:  # Weekend
            return True  # All day is peak on weekends
        else:
            return 18 <= hour < 23  # Peak hours on weekdays
    # Core Links
    elif link_category == 'core':
        if weekday < 5:  # Weekday
            return 8 <= hour < 18  # Peak hours on weekdays
        else:
            return False  # Off-peak on weekends
    # Edge Links
    elif link_category == 'edge':
        if weekday < 5:  # Weekday
            return 8 <= hour < 23  # Extended peak hours on weekdays
        else:
            return 12 <= hour < 23  # Peak hours on weekends
    else:
        return False  # Default to off-peak

# Function to calculate link metrics based on simulated time and state
def calculate_link_metrics(link):
    global current_simulated_time
    category = link['category']
    link_id = link['id']
    peak = is_peak_time(category, current_simulated_time)
    state = update_link_state(link_id, peak)
    config = LINK_CATEGORIES[category]

    # Determine time of day effect based on link category
    if peak:
        time_factor = 1.0  # Peak time
    else:
        time_factor = 0.5  # Off-peak time

    # Base metrics
    base_bw = config['max_bw'] * time_factor
    base_latency = config['min_latency'] / time_factor
    base_jitter = config['min_jitter'] / time_factor
    base_loss = config['min_loss'] * time_factor

    # State-based adjustments
    if state == 'Normal':
        state_factor = 1.0
    elif state == 'Congested':
        state_factor = 0.6
    elif state == 'Overloaded':
        state_factor = 0.3

    adjusted_bw = base_bw * state_factor
    adjusted_latency = base_latency / state_factor
    adjusted_jitter = base_jitter / state_factor
    adjusted_loss = base_loss * (1 / state_factor)

    # Ensure metrics are within min and max
    adjusted_bw = max(config['min_bw'], min(config['max_bw'], adjusted_bw))
    adjusted_latency = max(config['min_latency'], min(config['max_latency'], adjusted_latency))
    adjusted_jitter = max(config['min_jitter'], min(config['max_jitter'], adjusted_jitter))
    adjusted_loss = max(config['min_loss'], min(config['max_loss'], adjusted_loss))

    # Introduce randomness with normal distribution
    adjusted_bw += np.random.normal(0, (config['max_bw'] - config['min_bw']) * 0.05)
    adjusted_latency += np.random.normal(0, (config['max_latency'] - config['min_latency']) * 0.05)
    adjusted_jitter += np.random.normal(0, (config['max_jitter'] - config['min_jitter']) * 0.05)
    adjusted_loss += np.random.normal(0, (config['max_loss'] - config['min_loss']) * 0.05)

    # Ensure metrics are within min and max after randomness
    adjusted_bw = max(config['min_bw'], min(config['max_bw'], adjusted_bw))
    adjusted_latency = max(config['min_latency'], min(config['max_latency'], adjusted_latency))
    adjusted_jitter = max(config['min_jitter'], min(config['max_jitter'], adjusted_jitter))
    adjusted_loss = max(config['min_loss'], min(config['max_loss'], adjusted_loss))

    # Prepare metrics dictionary
    metrics = {
        'bw': adjusted_bw,
        'latency': adjusted_latency,
        'jitter': adjusted_jitter,
        'loss': adjusted_loss
    }

    return metrics

# Function to apply metrics to a link
def apply_metrics_to_link(link, metrics):
    payload = {
        'link': link['id'],
        'bw': metrics['bw'],
        'latency': metrics['latency'],
        'jitter': metrics['jitter'],
        'loss': metrics['loss']
    }

    # Send API request
    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=2  # seconds
        )
        if response.status_code == 200:
            pass  # Success; optionally log or print if needed
        else:
            print(f"[{current_simulated_time}] Failed to update link {link['id']}: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"[{current_simulated_time}] Exception when updating link {link['id']}: {e}")

# Main simulation loop
def run_simulation():
    global current_simulated_time

    simulated_time_seconds = 0
    total_steps = SIMULATION_DURATION // TIME_STEP

    while simulated_time_seconds < SIMULATION_DURATION:
        # Calculate current simulated time
        current_simulated_time = SIMULATED_TIME_START + datetime.timedelta(seconds=simulated_time_seconds)

        # Print current simulated time
        print(f"\n\n######### Current simulated time #########: {current_simulated_time}")

        # Calculate and apply metrics for each link
        for link in LINKS:
            metrics = calculate_link_metrics(link)
            if simulated_time_seconds % 12000 == 0:
                apply_metrics_to_link(link, metrics)

        # Yield the current simulated time
        yield current_simulated_time

        # Advance simulated time
        simulated_time_seconds += TIME_STEP

# Entry point
if __name__ == "__main__":
    run_simulation()
