import os
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class BoxEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        self.size = size
        self.step_limit = 75
        self.n_boxes = 12
        self.n_obstacles = 6
        self.n_goals = 5
        self.box_block = True
        self.walls = False
        self.soft_obstacles = True

        self.use_obst = self.n_obstacles > 0 or self.walls

        # Set TILE_SIZE and window_size for rendering
        self.TILE_SIZE = 64
        self.window_size = self.size * self.TILE_SIZE

        # The observation space represents the grid with different layers
        # 0: agent, 1: goals, 2: boxes, 3: obstacles, 4: time remaining
        self.observation_space = spaces.Box(0, 1, (self.size, self.size, 4 + self.use_obst))
        self.action_space = spaces.Discrete(4)

        # Mapping of actions to vectors for grid movement
        # Note: (1, 0) is down in NumPy array indexing (row-first)
        self._action_to_direction = {
            Actions.down.value: np.array([1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.right.value: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.textures = {}
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Box Env")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.load_textures()

    def _seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        obs = np.zeros((self.size, self.size, 4 + self.use_obst), dtype=np.float32)
        x, y = self._agent_location
        obs[x, y, 0] = 1.0

        for gx, gy in self._target_location:
            obs[gx, gy, 1] = 1.0

        for bx, by in self._boxes:
            obs[bx, by, 2] = 1.0

        if self.use_obst:
            for ox, oy in self._obstacles:
                obs[ox, oy, 3] = 1.0

        obs[:, :, 3 + self.use_obst].fill((self.step_limit - self.steps_taken) / self.step_limit)

        return obs

    def _get_info(self):
        boxes_on_goals = self.boxes_on_goals_count
        boxes_left = len(self._boxes)

        return {
            "boxes_on_goals": boxes_on_goals,
            "boxes_left": boxes_left
        }

    def _is_in_grid(self, point):
        return (0 <= point[0] < self.size) and (0 <= point[1] < self.size)

    def _is_on_edge(self, point):
        return (point[0] == 0) or (point[1] == 0) or (point[0] == self.size - 1) or (point[1] == self.size - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.edge_boxes = 0
        
        self.boxes_on_goals_count = 0

        grid_size = self.size - 2 * self.walls
        locs = self.np_random.choice(grid_size**2, 1 + self.n_goals + self.n_boxes + self.n_obstacles, replace=False)
        xs, ys = np.unravel_index(locs, (grid_size, grid_size))
        xs += self.walls
        ys += self.walls

        self._agent_location = np.array([xs[0], ys[0]])
        self._target_location = [np.array([xs[i], ys[i]]) for i in range(1, 1 + self.n_goals)]
        self._boxes = [np.array([xs[i], ys[i]]) for i in range(1 + self.n_goals, 1 + self.n_goals + self.n_boxes)]
        self._obstacles = (
            [np.array([xs[i], ys[i]]) for i in range(1 + self.n_goals + self.n_boxes, len(xs))]
            if self.use_obst
            else []
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        vec = self._action_to_direction[action]
        old_pos = self._agent_location.copy()
        next_pos = old_pos + vec
        final_pos = old_pos.copy() # Store the final valid position

        reward = -0.01
        terminated = False
        
        box_idx = None
        for i, box in enumerate(self._boxes):
            if np.array_equal(next_pos, box):
                box_idx = i
                break

        if not self._is_in_grid(next_pos):
            reward = -1.0
            terminated = True
        elif self.use_obst and any(np.array_equal(next_pos, obs) for obs in self._obstacles):
            if not self.soft_obstacles:
                reward = -0.1
                final_pos = old_pos  # Stay in place
            else:
                reward = -0.2
                final_pos = next_pos
        elif box_idx is not None:
            push_pos = next_pos + vec
            if not self._is_in_grid(push_pos):
                self._boxes.pop(box_idx)
                # `self.boxes_left` is now calculated from the list length
                self.edge_boxes += 1
                reward = -0.1
                final_pos = next_pos
            elif any(np.array_equal(push_pos, box) for i, box in enumerate(self._boxes) if i != box_idx):
                if self.box_block:
                    reward = -0.1
                    final_pos = old_pos
                else:
                    target_box_idx = [i for i, b in enumerate(self._boxes) if np.array_equal(push_pos, b)][0]
                    self._boxes.pop(target_box_idx)
                    if box_idx > target_box_idx:
                        box_idx -= 1
                    self._boxes.pop(box_idx)
                    # `self.boxes_left` is now calculated from the list length
                    reward = -0.1
                    final_pos = next_pos
            elif self.use_obst and any(np.array_equal(push_pos, obs) for obs in self._obstacles):
                if self.soft_obstacles:
                    reward = -0.2
                    self._boxes.pop(box_idx)
                    self._boxes.append(push_pos)
                    final_pos = next_pos
                else:
                    reward = -0.1
                    final_pos = old_pos
            elif any(np.array_equal(push_pos, goal) for goal in self._target_location):
                self._boxes.pop(box_idx)
                # Correctly increment the counter for boxes on goals
                self.boxes_on_goals_count += 1
                reward = 1.0
                final_pos = next_pos
            else:
                self._boxes[box_idx] = push_pos
                if self._is_on_edge(push_pos):
                    self.edge_boxes += 1
                final_pos = next_pos
        else:
            final_pos = next_pos

        self._agent_location = final_pos
        self.steps_taken += 1

        # Now checks the correct `boxes_left` count for termination
        if self.steps_taken >= self.step_limit or len(self._boxes) <= self.edge_boxes:
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def clone_state(self):
        return {
            "agent": np.copy(self._agent_location),
            "targets": [np.copy(t) for t in self._target_location],
            "boxes": [np.copy(b) for b in self._boxes],
            "obstacles": [np.copy(o) for o in self._obstacles],
            "steps_taken": self.steps_taken,
            "edge_boxes": self.edge_boxes,
            "boxes_on_goals_count": self.boxes_on_goals_count,
            "rng_state": self.np_random.bit_generator.state,
        }

    def restore_state(self, state):
        self._agent_location = np.copy(state["agent"])
        self._target_location = [np.copy(t) for t in state["targets"]]
        self._boxes = [np.copy(b) for b in state["boxes"]]
        self._obstacles = [np.copy(o) for o in state["obstacles"]]
        self.steps_taken = state["steps_taken"]
        self.edge_boxes = state["edge_boxes"]
        self.boxes_on_goals_count = state["boxes_on_goals_count"]
        self.np_random.bit_generator.state = state["rng_state"]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def load_textures(self):
        path = os.path.join(os.path.dirname(__file__), "resources", "textures")
        self.textures = {}
        texture_files = {
            "agent": "player.png",
            "box": "box.png",
            "goal": "goal.png",
            "wall": "wall.png",
            "floor": "land.png",
            "obstacle": "obstacle.png"
        }

        for key, fname in texture_files.items():
            img = pygame.image.load(os.path.join(path, fname)).convert_alpha()
            self.textures[key] = pygame.transform.scale(img, (self.TILE_SIZE, self.TILE_SIZE))

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        
        # Draw the floor
        for x in range(self.size):
            for y in range(self.size):
                canvas.blit(self.textures["floor"], (y * self.TILE_SIZE, x * self.TILE_SIZE))

        # Draw goals, walls, boxes, and agent in a specific order
        for goal_pos in self._target_location:
            x, y = goal_pos
            canvas.blit(self.textures["goal"], (y * self.TILE_SIZE, x * self.TILE_SIZE))
        
        if self.use_obst:
            for obs_pos in self._obstacles:
                x, y = obs_pos
                canvas.blit(self.textures["obstacle"], (y * self.TILE_SIZE, x * self.TILE_SIZE))

        for box_pos in self._boxes:
            x, y = box_pos
            canvas.blit(self.textures["box"], (y * self.TILE_SIZE, x * self.TILE_SIZE))
        
        x, y = self._agent_location
        canvas.blit(self.textures["agent"], (y * self.TILE_SIZE, x * self.TILE_SIZE))

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, x * self.TILE_SIZE), (self.window_size, x * self.TILE_SIZE), width=1)
            pygame.draw.line(canvas, (0, 0, 0), (x * self.TILE_SIZE, 0), (x * self.TILE_SIZE, self.window_size), width=1)

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
