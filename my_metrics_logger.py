from abc import ABC



from rlgym_ppo.util import reporting
from rlgym_sim.utils.gamestates import GameState
import numpy as np
from rlgym_ppo.util import reporting
from rlgym_sim import gym



class myMetricsLogger(ABC):
    def __init__(self):
        self.last_ball_position = None  # Initialize last ball position
        self.time_elapsed = 0
        self.episodes_elapsed = 0
        self.average_episode_length = 0
        self.episode_end = False
    def collect_metrics(self, game_state: GameState) -> np.ndarray:
        self.time_elapsed += 1

        # Initialize unraveled as an empty list
        unraveled = []
        current_ball_position = game_state.ball.position
        if self.last_ball_position is not None:
            distance_moved = np.linalg.norm(current_ball_position - self.last_ball_position)
        else:
            distance_moved = 0
        #print("distance_moved", distance_moved)
        # Update last ball position
        self.last_ball_position = current_ball_position

        self.episode_end = distance_moved > 50
        #print("episode_end", episode_end)
        self.episodes_elapsed += self.episode_end
        self.average_episode_length = self.time_elapsed/(self.episodes_elapsed+1)
        xy_dist = game_state.ball.position - game_state.players[0].car_data.position


        unraveled.append(self.episode_end)
        unraveled.append(self.average_episode_length)
        unraveled.append(self.time_elapsed)
        unraveled.append(self.episodes_elapsed)
        return np.asarray(unraveled).astype(np.float32)

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if wandb_run is None:
            return

        average_episode_length = collected_metrics[0][1]
        time_elapsed = collected_metrics[0][2]
        episodes_elapsed = collected_metrics[0][3]
        report = {
            'time_elapsed': time_elapsed,
            'episodes_elapsed': episodes_elapsed,
            'average_episode_length': average_episode_length,
            'Episode TimeCT': len(collected_metrics[0])
        }
        print("cumulative_timesteps", cumulative_timesteps)
        #report metrics to wandb

        #print("reporting, ", time_elapsed, episodes_elapsed, average_episode_length)
        wandb_run.log(report)
        pass



    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        raise NotImplementedError

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        raise NotImplementedError