from enum import Enum
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pygame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sklearn.metrics.pairwise import manhattan_distances

from train_agent import train_agent
from qlearner import QLearner

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size=5, min_m=5, max_m=10):
        self.window_size = 512
        self.size = size
        self.observation_space = spaces.Dict(
            {
                "agent" : spaces.Box(0, size-1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.reward = 1

        self.walls = []
        self.bad_tiles = []
        self.special_tiles = []

        self.special_reward_a = False
        self.special_reward_b = False

        self.wall_correct = False
        self.bad_correct = False

        self.min_moves_to_win = min_m
        self.max_moves_to_win = max_m

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _manhattan_distance(self, A, B, min_value, max_value=np.inf):

        dist = manhattan_distances([A], [B]) [0] [0]

        if dist >= min_value and dist <= max_value:
            return True
        else:
            return False

    def _get_neighbors(self, pos):
        pos_x = pos[0]
        pos_y = pos[1]

        indices = [(pos_x-1, pos_y-1), (pos_x-1, pos_y), (pos_x-1, pos_y+1),
                    (pos_x, pos_y-1), (pos_x, pos_y+1),
                   (pos_x+1, pos_y-1), (pos_x+1, pos_y),(pos_x+1, pos_y+1)]
        indices = list(filter(lambda idx : True if (0 <= idx[0] <= self.size-1) and (0 <= idx[1] <= self.size-1)
                                                else False, indices))

        return indices

    def _check_placing(self):

        start = self._agent_location
        end = self._target_location

        visited = set()
        queue = deque([(start, [start])])  

        while queue:
            current, path = queue.popleft()
            if tuple(current) == tuple(end):
                return path

            for move in self._action_to_direction.values():
                neighbor = tuple(np.clip(np.array(current) + move, 0, self.size - 1))
                if neighbor in visited or neighbor in self.walls or neighbor in self.bad_tiles:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

        return None  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset = False
        correct_match = False
        iter = 0

        while not correct_match:
            reset = False
            iter += 1
            self.walls = []
            self.bad_tiles = []
            self.special_tiles = []

            if iter > 100:
                raise Exception("No suitable placing variant found")

            self.available_pos = [(i,j) for j in range(self.size) for i in range(self.size)]
            # print("Iteration ", iter, "Available: ", self.available_pos)
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._target_location = self._agent_location

            # SETTING TARGET AND AGENT
            tries = 0
            while not self._manhattan_distance(self._target_location, self._agent_location, self.min_moves_to_win,
                                               self.max_moves_to_win):
                tries += 1
                if tries == 100:
                    reset = True
                    break

                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

            if not reset:
                self.available_pos.remove(tuple(self._agent_location))
                self.available_pos.remove(tuple(self._target_location))

            # SETTING WALLS
            tries = 0
            while(len(self.walls) < 5 and not reset):
                tries += 1
                if tries == 100:
                    reset = True
                    break


                wall_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
                if wall_pos in self.available_pos and not self.wall_correct \
                        and self._manhattan_distance(wall_pos, self._agent_location, self.min_moves_to_win-3,
                                                     self.min_moves_to_win) \
                        and self._manhattan_distance(wall_pos, self._target_location, self.min_moves_to_win-3,
                                                     self.min_moves_to_win):
                    self.walls.append(wall_pos)
                    self.available_pos.remove(wall_pos)
                    self.wall_correct = True
                elif not self.wall_correct:
                    continue
                elif wall_pos in self.available_pos and not list(set(self._get_neighbors(wall_pos)) & set(self.walls)) :
                    self.walls.append(wall_pos)
                    self.available_pos.remove(wall_pos)


            # SETTING BAD_TILES
            tries = 0
            num_bad = np.random.randint(3, 6)
            while (len(self.bad_tiles) < num_bad and not reset):
                tries += 1
                if tries == 100:
                    reset = True
                    break

                bad_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))

                if bad_pos in self.available_pos and not self.bad_correct \
                        and self._manhattan_distance(bad_pos, self._agent_location, self.min_moves_to_win-3,
                                                     self.min_moves_to_win)\
                        and self._manhattan_distance(bad_pos, self._target_location, self.min_moves_to_win-3,
                                                     self.min_moves_to_win):
                    self.bad_tiles.append(bad_pos)
                    self.available_pos.remove(bad_pos)
                    self.bad_correct = True
                elif not self.bad_correct:
                    continue
                elif bad_pos in self.available_pos and not list(set(self._get_neighbors(bad_pos)) & set(self.bad_tiles)) :
                    self.bad_tiles.append(bad_pos)
                    self.available_pos.remove(bad_pos)


            # SETTING SPECIALS
            tries = 0
            while(len(self.special_tiles) < 2 and not reset):
                tries += 1
                if tries == 100:
                    reset = True
                    break
                special_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
                if special_pos in self.available_pos:
                    self.special_tiles.append(special_pos)
                    self.available_pos.remove(special_pos)


            if not reset:
                if self._check_placing() is not None:
                    correct_match = True
                else:
                    reset = True


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        special_event = ""

        random_v = np.random.rand()
        if  0.9 < random_v < 1.0 and tuple(self._agent_location + direction) not in self.walls:
            direction = 2 * direction
            special_event += " Double move "

        possible_move = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if tuple(possible_move) not in self.walls:
            self._agent_location = possible_move

        terminated = np.array_equal(self._agent_location, self._target_location)

        if tuple(self._agent_location) in self.bad_tiles:
            self.reward -= 10
            terminated = True
            special_event += " Bad tile (-10) "

        elif tuple(self._agent_location) == tuple(self.special_tiles[0]):
            if not self.special_reward_a:
                self.reward += 10
                self.special_reward_a = True
                special_event += " Reward A (+10) "
            else:
                special_event += " Reward A (reward was used) "
        elif tuple(self._agent_location) == tuple(self.special_tiles[1]):
            if not self.special_reward_b:
                self.reward += 10
                self.special_reward_b = True
                terminated = True
                special_event += " Reward B (+10) "
            else:
                special_event += " Reward B (reward was used) "

        if terminated:
            if tuple(self._agent_location) not in self.bad_tiles and tuple(self._agent_location) not in self.special_tiles:
                self.reward = 100
        else:
            self.reward -= 0.1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward, terminated, False, info, special_event

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        circle = pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for wall_pos in self.walls:
            pygame.draw.rect(
                canvas,
                (0,0,0),
                pygame.Rect(
                    np.array(wall_pos) * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),
            )

        for bad_pos in self.bad_tiles:
            pygame.draw.rect(
                canvas,
                (105,105,105),
                pygame.Rect(
                    np.array(bad_pos) * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),
            )
        rect_a = pygame.draw.rect(
            canvas,
            (255,215,0),
            pygame.Rect(
                np.array(self.special_tiles[0]) * pix_square_size,
                (pix_square_size, pix_square_size),
            ),
        )

        text_obj_a = pygame.font.SysFont('Arial', 50).render('A', True, (0,0,0))
        text_rect_a = text_obj_a.get_rect(center=rect_a.center)
        canvas.blit(text_obj_a, text_rect_a)

        rect_b = pygame.draw.rect(
            canvas,
            (255,20,147),
            pygame.Rect(
                np.array(self.special_tiles[1]) * pix_square_size,
                (pix_square_size, pix_square_size),
            ),
        )
        text_obj_b = pygame.font.SysFont('Arial', 50).render('B', True, (0, 0, 0))
        text_rect_b = text_obj_b.get_rect(center=rect_b.center)

        canvas.blit(text_obj_b, text_rect_b)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# render_mode= 'human' # None for training or 'human' for visualization
render_mode = None
# env = GridEnv(render_mode=render_mode)
env = GridEnv(render_mode=render_mode, min_m=11, max_m=float('inf'), size=8)
obs, _ = env.reset()
visualize = False
#hyperparameters
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 2000
test_episodes = 100
max_steps = 100

alpha = 0.8
discount_factor = 0.7

n_bins = 8
algorithm = 'qlearn' # 'ppo' or 'qlearn'


if render_mode == None:
    if algorithm == 'qlearn':
        agent = QLearner(env, n_bins, alpha, discount_factor, max_epsilon, min_epsilon, decay, adaptive_mode=True, adaptive_binning=True)
        training_rewards, epsilons, train_episodes, states = train_agent(agent, max_steps, diff=0.001)
    elif algorithm == 'ppo': # NOT ENDED
        agent = PPO("MultiInputPolicy", env, verbose=1)
        agent.learn(total_timesteps=25000)
        agent.save("ppo_cartpole")

        del agent  # remove to demonstrate saving and loading

        agent = PPO.load("ppo_cartpole")

        obs = env.reset()
        while True:
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # env.render("human")
    else:
        pass
elif render_mode == 'human':
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info, special_event = env.step(action)
        print(f"Action: {action}, Reward: {reward : .1f}, Done: {done}, Special event: {special_event}")

env.close()


if visualize and render_mode is None:
    x = range(train_episodes)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training')
    plt.grid()
    plt.show()

    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title("Epsilon for episode")
    plt.grid()
    plt.show()

    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    window_size = 10

    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(training_rewards, window_size=window_size), color='blue', label='Smoothed (10 ep MA)')
    plt.title("Learning Curve: Total Reward per Episode (with average window = {})".format(window_size))
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

