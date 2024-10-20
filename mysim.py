import numpy as np
import rlgym_sim
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.state_setters.wrappers.state_wrapper import StateWrapper
from my_continuous_act import ContinuousAction


class BallTouchGroundCondition(TerminalCondition):
    def __init__(self, min_ball_height=110):
        super().__init__()
        self.min_ball_height = min_ball_height

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        # Check if the ball's z-position is less than or equal to the minimum height
        if current_state.ball.position[2] <= self.min_ball_height:
            return True
        else:
            return False

class CustomStateSetter(StateSetter):
    def __init__(self, ball_height_above_car= 100):
        super().__init__()
        self.ball_height_above_car = ball_height_above_car


    def reset(self, state_wrapper: StateWrapper):
        base_velocity_x = np.random.normal(loc=200, scale=120)
        velocity_y = np.random.normal(loc=50, scale=30)
        pos_x = np.random.normal(loc=0, scale=1000)
        pos_y = np.random.normal(loc=0, scale=1200)
        state_wrapper.cars[0].boost = 100
        # Set ball position and velocity
        ball_state = state_wrapper.ball
        ball_state.position[2] = self.ball_height_above_car + 93
        ball_state.position[0] = pos_x
        ball_state.position[1] = pos_y
        ball_state.linear_velocity[0] = base_velocity_x

        if len(state_wrapper.cars) > 0:
            car_state = state_wrapper.cars[0]
            car_state.position[0] = pos_x
            car_state.position[1] = pos_y
            #car_state.speed = base_velocity_x
            car_state.linear_velocity[0] = base_velocity_x
            #car_state.linear_velocity[1] = velocity_y


class SimpleDribbleObs(ObsBuilder):

    def __init__(self):
        super().__init__()
        self.ball_touch_list = [0, 0, 0]

        #initialize velocity list as list of vectors
        self.ball_velocity_list = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        ball = state.ball
        car = player.car_data

        # Calculate relative position and velocity of ball to car
        rel_pos = ball.position - car.position
        rel_vel = ball.linear_velocity - car.linear_velocity

        # Calculate the rotation to apply based on the car's yaw
        cos_yaw = np.cos(car.yaw())
        sin_yaw = np.sin(car.yaw())

        # Apply rotation to relative position
        rel_pos_rotated = np.array([
            cos_yaw * rel_pos[0] + sin_yaw * rel_pos[1],
            -sin_yaw * rel_pos[0] + cos_yaw * rel_pos[1],
            rel_pos[2]
        ])

        # Apply rotation to relative velocity
        rel_vel_rotated = np.array([
            cos_yaw * rel_vel[0] + sin_yaw * rel_vel[1],
            -sin_yaw * rel_vel[0] + cos_yaw * rel_vel[1],
            rel_vel[2]
        ])
        # Get ball's angular velocity and rotate it
        ball_spin = ball.angular_velocity
        ball_spin_rotated = np.array([
            cos_yaw * ball_spin[0] + sin_yaw * ball_spin[1],
            -sin_yaw * ball_spin[0] + cos_yaw * ball_spin[1],
            ball_spin[2]
        ])

        # Calculate the intersection of the car's direction vector with the wall
        car_direction = np.array([np.cos(car.yaw()), np.sin(car.yaw())])
        x_wall = 4096 if car_direction[0] > 0 else -4096
        y_wall = 5120 if car_direction[1] > 0 else -5120

        t_x = (x_wall - car.position[0]) / (car_direction[0] if car_direction[0] != 0 else (car_direction[0] + 0.01))
        t_y = (y_wall - car.position[1]) / (car_direction[1] if car_direction[1] != 0 else (car_direction[1] + 0.01))

        t = min(t_x, t_y)
        intersection = car.position[:2] + t * car_direction

        # Calculate the L2 distance from the car to the intersection point
        distance_to_wall = np.linalg.norm(intersection - car.position[:2])

        # Calculate the intersection of the car's direction plus a small offset vector with the wall
        car_direction_offset = np.array([np.cos(car.yaw() + np.pi/14), np.sin(car.yaw()+np.pi/14)])

        x_wall = 4096 if car_direction_offset[0] > 0 else -4096
        y_wall = 5120 if car_direction_offset[1] > 0 else -5120

        t_x = (x_wall - car.position[0]) / (car_direction_offset[0] if car_direction_offset[0] != 0 else (car_direction_offset[0] + 0.01))
        t_y = (y_wall - car.position[1]) / (car_direction_offset[1] if car_direction_offset[1] != 0 else (car_direction_offset[1] + 0.01))

        t = min(t_x, t_y)
        intersection_offset = car.position[:2] + t * car_direction_offset

        # Calculate the L2 distance from the car to the intersection point
        distance_to_wall_offset = np.linalg.norm(intersection_offset - car.position[:2])

        if distance_to_wall_offset < distance_to_wall:
            wall_bump_feature = 1
        else:
            wall_bump_feature = -1

        self.ball_velocity_list.append(ball.linear_velocity)

        # Calculate the rel_acc_rotated of the ball
        rel_acc_rotated = self.ball_velocity_list[1] - self.ball_velocity_list[0]
        self.ball_velocity_list.pop(0)

        #rotate rel_acc_rotated to match car
        cos_yaw = np.cos(car.yaw())
        sin_yaw = np.sin(car.yaw())
        rel_acc_rotated = np.array([
            cos_yaw * rel_acc_rotated[0] + sin_yaw * rel_acc_rotated[1],
            -sin_yaw * rel_acc_rotated[0] + cos_yaw * rel_acc_rotated[1],
            rel_acc_rotated[2]
        ])

        if player.ball_touched:
            self.ball_touch_list.append(1)
        else:
            self.ball_touch_list.append(0)
        self.ball_touch_list.pop(0)

        car_speed = np.linalg.norm(car.linear_velocity)
        obs = np.array([
            wall_bump_feature,
            rel_acc_rotated[0],
            rel_acc_rotated[1],
            rel_acc_rotated[2],
            self.ball_touch_list[0],
            self.ball_touch_list[1],
            self.ball_touch_list[2],
            car_speed/200,
            rel_pos_rotated[0] / 10,  # Relative position x of ball to car, now in car's forward direction
            rel_pos_rotated[1] / 10,  # Relative position y of ball to car, right of the car
            rel_pos_rotated[2] / 10,  # Relative position z of ball to car, up from the car
            rel_vel_rotated[0] / 10,  # Relative velocity x, forward direction
            rel_vel_rotated[1] / 10,  # Relative velocity y, rightward direction
            rel_vel_rotated[2] / 10,  # Relative velocity z, upward direction
            distance_to_wall/2000,
        ]).flatten()
        return obs


class stayNearMiddleReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0  # Initialize a counter to track the number of steps
        self.consecutive_touches = 0
    def reset(self, initial_state: GameState):
        self.time_elapsed = 0  # Reset the counter at the start of each episode
        self.consecutive_touches = 0
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.time_elapsed += 1
        ball = state.ball
        relative_pos_xy = player.car_data.position[:2] - ball.position[:2]  # Only take the x and y components
        xy_dist = np.linalg.norm(relative_pos_xy)
        reward = 30*self.time_elapsed/xy_dist

        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
        return -xy_dist/70

class consecutiveTouches(RewardFunction):
        def __init__(self):
            super().__init__()
            self.time_elapsed = 0  # Initialize a counter to track the number of steps
            self.consecutive_touches = 0
            self.cum_reward = 0
            self.episode_length = 0  # Add this line to track episode length

        def reset(self, initial_state: GameState):
            self.time_elapsed = 0  # Reset the counter at the start of each episode
            self.consecutive_touches = 0
            self.cum_reward = 0
            self.episode_length = 0  # Reset

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            self.time_elapsed += 1
            self.episode_length += 1  # Increment episode length

            ball = state.ball
            ball_height = ball.position[2]
            reward = 0

            if player.ball_touched:
                self.consecutive_touches += 1

                reward = self.consecutive_touches * 40

            else:
                if self.consecutive_touches > 0:
                    self.consecutive_touches -= 1

            if self.consecutive_touches >= 8:
                self.consecutive_touches = 7

            # print("dribble reward: ", reward)
            return reward/40

        def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
            return -xy_dist/70

class ezDribbleReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0  # Initialize a counter to track the number of steps
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Add this line to track episode length
    def reset(self, initial_state: GameState):
        self.time_elapsed = 0  # Reset the counter at the start of each episode
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Reset
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.time_elapsed += 1
        self.episode_length += 1  # Increment episode length

        ball = state.ball
        ball_height = ball.position[2]
        reward = 0
        relative_pos_xy = player.car_data.position[:2] - ball.position[:2]  # Only take the x and y components
        xy_dist = np.linalg.norm(relative_pos_xy)
        reward += 119.683 * np.exp(((xy_dist/20)-0.8)**2 * -1)


        
        return reward/120

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
        return -xy_dist/70

class velocityWithBall(RewardFunction):
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0  # Initialize a counter to track the number of steps
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Add this line to track episode length

    def reset(self, initial_state: GameState):
        self.time_elapsed = 0  # Reset the counter at the start of each episode
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Reset

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.time_elapsed += 1
        self.episode_length += 1

        ball = state.ball
        reward = 0
        ball_vel = ball.linear_velocity
        player_vel = player.car_data.linear_velocity
        reward -= np.linalg.norm(player_vel - ball_vel)

        #print("velWithBall reward: ", reward)
        return reward/355

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
        return -xy_dist/70

class ballVelocity(RewardFunction):
        def __init__(self):
            super().__init__()
            self.time_elapsed = 0  # Initialize a counter to track the number of steps
            self.consecutive_touches = 0
            self.cum_reward = 0
            self.episode_length = 0  # Add this line to track episode length

        def reset(self, initial_state: GameState):
            self.time_elapsed = 0  # Reset the counter at the start of each episode
            self.consecutive_touches = 0
            self.cum_reward = 0
            self.episode_length = 0  # Reset

        def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            self.time_elapsed += 1
            self.episode_length += 1

            reward = np.linalg.norm(state.ball.linear_velocity)

            #print("ball vel reward: ", reward)
            #print("ball vel reward: ", reward)
            return reward/150

        def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
            xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
            return -xy_dist/70


class noSlide(RewardFunction):
    def __init__(self):
        super().__init__()
        self.time_elapsed = 0  # Initialize a counter to track the number of steps
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Add this line to track episode length

    def reset(self, initial_state: GameState):
        self.time_elapsed = 0  # Reset the counter at the start of each episode
        self.consecutive_touches = 0
        self.cum_reward = 0
        self.episode_length = 0  # Reset

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.time_elapsed += 1
        self.episode_length += 1

        #penalize for using powerslide action
        reward = 0
        if previous_action[1] == 0:
            reward -= 1
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        xy_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
        return -xy_dist / 70

def create_rlgym_sim_env():
    combined_reward = CombinedReward(reward_functions=(ezDribbleReward(), velocityWithBall(), ballVelocity(), consecutiveTouches(), VelocityPlayerToBallReward()), reward_weights=(1.5, 0.01, 0.01, 2.0, 0.01))
    env = rlgym_sim.make(
        #game_speed=1,
        state_setter=CustomStateSetter(),
        spawn_opponents=False,
        tick_skip=8,
        obs_builder=SimpleDribbleObs(),
        action_parser=ContinuousAction(),
        reward_fn=combined_reward,
        #use_injector=True,
        terminal_conditions=[BallTouchGroundCondition()],
        boost_consumption=0,
        gravity=1
    )
    return env





