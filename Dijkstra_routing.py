import heapq
import logging
import time
import requests
from Environment import SCIONEnvironment

class DijkstraRouter:
    def __init__(self, environment, metric="latency", check_interval=5):
        """
        Initialize the Dijkstra router with an environment and a specified metric.
        """
        self.environment = environment  # Instance of SCIONEnvironment
        self.metric = metric  # Metric to optimize: 'latency', 'hops', or 'bandwidth'
        self.check_interval = check_interval  # Time in seconds between re-evaluations
        self.current_best_path_id = None  # Store the current best path ID
        self.current_best_cost = float("inf")  # Store the cost of the current best path
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def calculate_cost(self, path):
        """
        Calculate the cost of a path based on the chosen metric.
        """
        if self.metric == "latency":
            return path.get("latency_ms", float("inf"))
        elif self.metric == "hops":
            return len(path.get("hops", []))
        elif self.metric == "bandwidth":
            return -path.get("bandwidth_mbps", 0)  # Higher bandwidth = lower cost
        else:
            raise ValueError("Invalid metric specified. Choose 'latency', 'hops', or 'bandwidth'.")

    def find_shortest_path(self):
        """
        Find the shortest path from the sender to the receiver using Dijkstra's algorithm.
        """
        paths = self.environment.get_paths()  # Fetch paths from the environment
        if not paths:
            self.logger.error("No paths available.")
            return None, None, float("inf")

        # Initialize priority queue
        pq = []
        for path_id, path_info in paths.items():
            cost = self.calculate_cost(path_info)
            heapq.heappush(pq, (cost, path_id))  # Push cost and path ID to the queue

        # Get the best path
        if pq:
            best_cost, best_path_id = heapq.heappop(pq)
            best_path = paths.get(best_path_id)
            return best_path, best_path_id, best_cost

        self.logger.error("No valid paths found.")
        return None, None, float("inf")

    def send_path_change_request(self, path_id):
        """
        Directly send a request to the environment to change the selected path.
        """
        try:
            response = requests.get(f"{self.environment.path_selection_url}{path_id}")
            response.raise_for_status()
            self.logger.info(f"Path changed successfully to path ID: {path_id}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to change path: {e}")

    def monitor_paths(self):
        """
        Continuously monitor paths and re-evaluate the best path.
        """
        while not self.environment.simulation_complited:
            best_path, best_path_id, best_cost = self.find_shortest_path()
            if best_path:
                if (
                    best_cost < self.current_best_cost
                    or best_path_id != self.current_best_path_id
                ):
                    self.logger.info(
                        f"Path updated! New best path based on {self.metric}: {best_path_id, best_path} with cost: {best_cost}"
                    )
                    self.current_best_path_id = best_path_id
                    self.current_best_cost = best_cost
                    self.send_path_change_request(best_path_id)  # Change the selected path
                else:
                    self.logger.info("Current path is still the best.")
            else:
                self.logger.error("Failed to find a new best path.")
            time.sleep(self.check_interval)

# Example usage
if __name__ == "__main__":
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/"
    )

    # Initialize the Dijkstra router with 'latency' as the optimization metric
    router = DijkstraRouter(environment=env, metric="latency", check_interval=5)
    router.monitor_paths()