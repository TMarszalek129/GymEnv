from enum import Enum
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics.pairwise import manhattan_distances

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size=5):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_pos = [(i,j) for j in range(self.size) for i in range(self.size)]
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location

        while not self._manhattan_distance(self._target_location, self._agent_location, 5, 10):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self.available_pos.remove(tuple(self._agent_location))
        self.available_pos.remove(tuple(self._target_location))

        while(len(self.walls) < 5):
            wall_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if wall_pos in self.available_pos:
                self.walls.append(wall_pos)
                self.available_pos.remove(wall_pos)

        num_bad = np.random.randint(3, 6)
        while (len(self.bad_tiles) < num_bad):
            bad_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if bad_pos in self.available_pos:
                self.bad_tiles.append(bad_pos)
                self.available_pos.remove(bad_pos)

        while(len(self.special_tiles) < 2):
            special_pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if special_pos in self.available_pos:
                self.special_tiles.append(special_pos)
                self.available_pos.remove(special_pos)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        possible_move = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if tuple(possible_move) not in self.walls:
            self._agent_location = possible_move

        terminated = np.array_equal(self._agent_location, self._target_location)
        special_event = ""
        if tuple(self._agent_location) in self.bad_tiles:
            self.reward -= 10
            terminated = True
            special_event = "Bad tile (-10)"

        elif tuple(self._agent_location) == tuple(self.special_tiles[0]):
            if not self.special_reward_a:
                self.reward += 10
                self.special_reward_a = True
                special_event = "Reward A (+10)"
            else:
                special_event = "Reward A (reward was used)"
        elif tuple(self._agent_location) == tuple(self.special_tiles[1]):
            if not self.special_reward_b:
                self.reward += 10
                self.special_reward_b = True
                terminated = True
                special_event = "Reward B (+10)"
            else:
                special_event = "Reward B (reward was used)"

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
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
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

        # Finally, add some gridlines
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
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

env = GridEnv(render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info, special_event = env.step(action)
    print(f"Action: {action}, Reward: {reward : .1f}, Done: {done}, Special event: {special_event}")


env.close()

        