# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image

from pidm_imitation.environment.toy_env.configs import (
    ToyEnvironmentActionConfig,
    ToyEnvironmentConfig,
    ToyEnvironmentObservationConfig,
    ToyEnvironmentRenderingConfig,
    ToyEnvironmentStateConfig,
)
from pidm_imitation.environment.toy_env.exogenous_noise_utils import (
    add_exogenous_noise_to_image,
)
from pidm_imitation.environment.toy_env.game_elements import (
    BlockSprite,
    Player,
    Ribbon,
    Wall,
    blocksprite_collide_with_all,
)
from pidm_imitation.environment.toy_env.toy_constants import (
    BACKGROUND_COLOR,
    NUM_TRIALS_RANDOM_SPAWN,
    RIBBON_COLOR,
)
from pidm_imitation.environment.toy_env.toy_types import ObservationType
from pidm_imitation.environment.toy_env.utils import toy_round
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


class ToyEnvironment(gym.Env, ABC):

    def __init__(
        self,
        config: ToyEnvironmentConfig | None,
    ):
        super().__init__()
        self.config: ToyEnvironmentConfig = config
        self.seed: int = config.seed
        self.room_size: np.ndarray = np.array(self.config.room_size)
        self.max_steps = self.config.max_steps
        self.observation_config: ToyEnvironmentObservationConfig = (
            config.observation_config
        )
        self.state_config: ToyEnvironmentStateConfig = config.state_config
        self.action_config: ToyEnvironmentActionConfig = config.action_config
        self.rendering_config: ToyEnvironmentRenderingConfig = config.rendering_config
        self.action_space: gym.spaces.Box = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        if self.seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(self.seed)

        # initial game
        pygame.init()
        self.screen = pygame.display.set_mode(tuple(self.room_size))
        pygame.display.set_caption("Toy Environment")
        self.screen.fill(BACKGROUND_COLOR)

        self.layout: str = self.config.layout
        self.positions = self.parse_layout_string()
        self.agent_positions = self.positions["agent"]
        self.wall_positions = self.positions["walls"]

        self.t = 0
        self.player: Player = Player(
            position=(0.0, 0.0), size=self.rendering_config.agent_size
        )
        self.walls: Set[Wall] = set()
        self.ribbons: Dict[str, Ribbon] = {}
        self._player_group: pygame.sprite.Group = pygame.sprite.Group(self.player)
        self._walls_group: pygame.sprite.Group | None = None
        self.last_exogenous_noise: np.ndarray = None

        if self.rendering_config.render_player_trace:
            self.add_ribbon(
                name="player_trace",
                width=self.rendering_config.trace_size,
                max_length=self.rendering_config.max_trace_length,
            )

    @abstractmethod
    def _get_feature_state_observation_dim(self) -> int:
        raise NotImplementedError

    def _build_feature_state_observation_space(self, num_values) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_values,), dtype=np.float32
        )

    def _build_image_observation_space(
        self, img_obs_size: Tuple[int, int]
    ) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(*img_obs_size, 3), dtype=np.uint8)

    def build_observation_space(self) -> gym.spaces.Space:
        if self.observation_config.observation_type == ObservationType.FEATURE_STATE:
            num_obs_values = self._get_feature_state_observation_dim()
            return self._build_feature_state_observation_space(num_obs_values)
        elif self.observation_config.observation_type == ObservationType.IMAGE_STATE:
            return self._build_image_observation_space(
                self.observation_config.img_obs_size
            )
        raise ValueError(
            f"Unsupported observation type {self.observation_config.observation_type}, must be one of "
            + f"{ObservationType.get_valid_values()}"
        )

    def add_ribbon(
        self,
        name: str,
        width: int = 1,
        color: Tuple[int, int, int] = RIBBON_COLOR,
        max_length: int = -1,
    ) -> Ribbon:
        ribbon = Ribbon(width, color=color, max_length=max_length)
        assert name not in self.ribbons, f"Ribbon with name {name} already exists"
        self.ribbons[name] = ribbon
        return ribbon

    @staticmethod
    def convert_position_to_room_size(
        pos: Tuple[float, float],
        layout_dims: Tuple[float, float],
        room_size: Tuple[float, float] | np.ndarray,
    ) -> np.ndarray:
        room_size = np.array(room_size)
        layout_height, layout_width = layout_dims
        layout_arr = np.array([layout_width, layout_height])
        if np.all(room_size == layout_arr):
            return np.array(pos)

        return np.ceil(np.array(pos) * room_size / layout_arr)

    @staticmethod
    def convert_wall_position_to_room_size(
        pos: Tuple[float, float, float, float],
        layout_dims: Tuple[float, float],
        room_size: Tuple[float, float] | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        layout_height, layout_width = layout_dims
        return (
            toy_round(pos[0] / layout_width * room_size[0], cast_to_int=True),
            toy_round(pos[1] / layout_height * room_size[1], cast_to_int=True),
            toy_round(pos[2] / layout_width * room_size[0], cast_to_int=True),
            toy_round(pos[3] / layout_height * room_size[1], cast_to_int=True),
        )

    @staticmethod
    def wall_has_vertical_neighbor(
        layout_rows: List[str], wall_position: Tuple[int, int]
    ):
        x, y = wall_position
        if y > 0 and layout_rows[y - 1][x] == "W":
            return True
        if y < len(layout_rows) - 1 and layout_rows[y + 1][x] == "W":
            return True
        return False

    @staticmethod
    def wall_has_horizontal_neighbor(
        layout_rows: List[str], wall_position: Tuple[int, int]
    ):
        x, y = wall_position
        if x > 0 and layout_rows[y][x - 1] == "W":
            return True
        if x < len(layout_rows[0]) - 1 and layout_rows[y][x + 1] == "W":
            return True
        return False

    def parse_layout_string(self) -> Dict[str, np.ndarray]:
        """Parse a map layout of the environment from a string.

        Legend:
        - A: agent candidates (at least one is required)
        - W: wall

        Example layout:
        WWWWWWWWWWW
        W         W
        W         W
        W         W
        W         W
        W    A    W
        W         W
        W         W
        W         W
        W         W
        WWWWWWWWWWW

        :param layout: string with the layout
        :return: dictionary mapping 'agent' and 'walls' to agent and goal positions (x, y)
        """
        layout_rows: List[str] = [row for row in self.layout.split("\n") if row.strip()]
        layout_dims = (len(layout_rows), len(layout_rows[0]))
        agent_positions = []
        wall_positions = []

        # Identify horizontal wall segments
        for y, row in enumerate(layout_rows):
            x_start = None
            for x, char in enumerate(row):
                if char == "W" and x_start is None:
                    x_start = x
                elif char != "W" and x_start is not None:
                    if (
                        x - x_start
                    ) > 1 or not ToyEnvironment.wall_has_vertical_neighbor(
                        layout_rows, (x_start, y)
                    ):
                        # do not add wall segments of width 1 that have a vertical neighbor --> include these only has
                        # part of a larger vertical wall
                        wall_positions.append(
                            (float(x_start), float(y), float(x - x_start), 1.0)
                        )
                    x_start = None
            if x_start is not None:
                wall_positions.append(
                    (float(x_start), float(y), float(len(row) - x_start), 1.0)
                )

        # Identify vertical wall segments
        for x in range(layout_dims[1]):
            y_start = None
            for y in range(layout_dims[0]):
                if layout_rows[y][x] == "W" and y_start is None:
                    y_start = y
                elif layout_rows[y][x] != "W" and y_start is not None:
                    if (y - y_start) > 1:
                        # only add vertical walls with height > 1
                        wall_positions.append(
                            (float(x), float(y_start), 1.0, float(y - y_start))
                        )
                    y_start = None
            if y_start is not None:
                wall_positions.append(
                    (float(x), float(y_start), 1.0, float(layout_dims[0] - y_start))
                )

        for y, row in enumerate(layout_rows):
            for x, char in enumerate(row):
                if char == "A":
                    agent_positions.append((float(x), float(y)))

        return {
            "agent": (
                np.stack(
                    [
                        ToyEnvironment.convert_position_to_room_size(
                            pos, layout_dims, self.room_size
                        )
                        for pos in agent_positions
                    ]
                )
                if agent_positions
                else np.array([])
            ),
            "walls": (
                np.stack(
                    [
                        ToyEnvironment.convert_wall_position_to_room_size(
                            pos, layout_dims, self.room_size
                        )
                        for pos in wall_positions
                    ]
                )
                if wall_positions
                else np.array([])
            ),
        }

    @property
    def objects(self) -> Set[BlockSprite]:
        objects: Set[BlockSprite] = self.walls  # type: ignore
        if self.player is not None:
            objects.add(self.player)
        return objects

    @property
    def player_group(self) -> pygame.sprite.Group:
        return self._player_group

    @property
    def walls_group(self) -> pygame.sprite.Group:
        return self._walls_group

    @property
    def objects_group(self) -> pygame.sprite.Group:
        return pygame.sprite.Group(self.objects)

    def _clip_position(
        self, position: np.ndarray, margin: np.ndarray | None = None
    ) -> np.ndarray:
        if margin is None:
            margin = np.zeros(2)
        assert len(position) == 2 and len(margin) == 2, "Position and margin must be 2D"
        assert np.all(margin >= 0), "Margin must be non-negative"
        return np.clip(position, margin, self.room_size - 1 - margin)

    def _get_random_position(self, margin: np.ndarray | None = None) -> np.ndarray:
        random_position = np.ceil(self._np_random.random(2) * self.room_size).astype(
            np.float32
        )
        return self._clip_position(random_position, margin)

    def _spawn_sprite_in_position(
        self,
        sprite: BlockSprite,
        position: np.ndarray,
    ):
        sprite.position = position
        sprite.previous_position = position

        objects_to_check = self.objects_group.copy()
        objects_to_check.remove(sprite)

        collision_dict = blocksprite_collide_with_all(sprite, objects_to_check)
        if collision_dict is not None:
            names = ",".join([o.name for o in collision_dict])
            raise ValueError(
                f"BlockSprite {sprite.name} collides with other objects: {names}"
            )

    def _spawn_sprite_in_random_position(
        self, sprite: BlockSprite, num_trials: int = NUM_TRIALS_RANDOM_SPAWN
    ):
        sprite.position = self._get_random_position()
        sprite.previous_position = sprite.position

        objects_to_check = self.objects_group.copy()
        objects_to_check.remove(sprite)

        trial = 0
        while (
            blocksprite_collide_with_all(sprite, objects_to_check) is not None
            and trial < num_trials
        ):
            sprite.position = self._get_random_position()
            sprite.previous_position = sprite.position
            trial += 1

        if trial >= num_trials:
            raise ValueError(
                f"Could not find a valid spawn position for sprite {sprite.name}"
            )

    def _move_player(self):
        self.player.update(velocity_scale=self.action_config.velocity_scale)
        previous_position = self.player.previous_position
        self.player.position = np.round(self.player.position, decimals=3)
        self.player.position = self._clip_position(self.player.position)
        self.player.previous_position = previous_position

    def _resolve_collisions(self) -> bool:
        for obj in self.objects:
            if isinstance(obj, Wall):
                continue

            retries = 0
            while (
                collided_walls := blocksprite_collide_with_all(obj, self.walls_group)
            ) is not None:
                assert all(isinstance(wall, Wall) for wall in collided_walls)
                self._bounce_off_walls(obj, collided_walls)
                retries += 1
                if retries >= 100:
                    # fallback to previous position
                    previous_position = obj.previous_position
                    obj.position = obj.previous_position
                    obj.previous_position = previous_position
                    return True
        return True

    def _bounce(
        self, vector: np.ndarray, hit: pygame.Rect
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve collision of a block sprite with a part of a wall and return the bounce
        vector and the amount in each dimension it has to move to avoid the wall."""
        dx, dy = vector
        movement = np.zeros(2)
        if hit.width > hit.height:
            # hit is mostly horizontal, so bounce vertically back
            dy = -dy
            dx = 0
            movement[1] = hit.height
        else:
            # bounce horizontally
            dx = -dx
            dy = 0
            movement[0] = hit.width
        return (np.array([dx, dy]), movement)

    def _bounce_off_walls(self, other: BlockSprite, walls: List[BlockSprite]) -> None:
        """Resolve collision of wall blocks with other object. The other object could be any entity in the environment.
        We compute the combined boundary of the wall blocks so the correction doesn't just move the object into another
        wall block.  We treat the list of walls as a single unified mass.

        Note: Our collision resolution only triggers if the object ever collides with the wall --> for sufficiently
        large movement speed, objects could "jump over" walls since from one step to another they move from one side
        to the other side of the wall without colliding with it. We assume that the movement speed is small enough
        that this does not happen.

        :param other: The other object to resolve collision with.
        :param walls: The list of wall blocks to resolve collision with.
        """
        vector = other.position - other.previous_position

        # compute the combined intersection of all overlapping regions, this is the region to avoid!
        # if we hit a straight will this will be long and thin, if we hit a corner it will be a box.
        combined_bounce = np.zeros(2)
        combined_movements = []
        for wall in walls:
            clip = other.rect.clip(wall.rect)
            bounce, movement = self._bounce(vector, clip)
            combined_bounce += bounce
            combined_movements.append(movement)

        # get the elementwise max of the movements.
        movement = np.max(combined_movements, axis=0) + 1.0
        # and direct them in the direction of our bounce vector.
        movement *= np.sign(combined_bounce)

        previous_position = other.previous_position
        other.position += movement
        other.previous_position = previous_position

    def spawn_wall_in_position(
        self, name: str, position: np.ndarray, size: Tuple[int, int]
    ) -> Wall:
        wall = Wall(position, size, name=name)
        if self.player is not None and wall.collide_with(self.player):
            raise ValueError(f"Wall {wall.name} collides with agent")
        self.walls.add(wall)
        return wall

    def spawn_agent_in_position(self, agent: Player, position: np.ndarray):
        self._spawn_sprite_in_position(agent, position)

    def spawn_agent_in_random_position(
        self, agent: Player, num_trials: int = NUM_TRIALS_RANDOM_SPAWN
    ):
        self._spawn_sprite_in_random_position(agent, num_trials=num_trials)

    def _spawn_walls(self):
        for i, (wall_x, wall_y, wall_width, wall_height) in enumerate(
            self.wall_positions
        ):
            wall_corner = np.array([wall_x, wall_y])
            wall_dims = np.array([wall_width, wall_height])
            wall_center = (wall_corner + wall_dims / 2).astype(int)
            self.spawn_wall_in_position(
                name=f"wall_{i}",
                position=wall_center,
                size=(wall_width, wall_height),
            )
        self._walls_group = pygame.sprite.Group(self.walls)

    def _spawn_agent(self):
        agent = Player(
            position=self.agent_positions[0], size=self.rendering_config.agent_size
        )
        # select random agent position from candidates
        agent_position = self._np_random.choice(self.agent_positions)
        self.spawn_agent_in_position(agent, position=agent_position)
        self.player = agent
        self._player_group = pygame.sprite.Group(self.player)

    def _clear_env(self):
        self.player = None
        self.walls.clear()
        self.t = 0
        self.last_exogenous_noise = None

    def _spawn_all(self):
        self._spawn_walls()
        self._spawn_agent()

    def _reset_rendering(self):
        for ribbon in self.ribbons.values():
            ribbon.reset()
        self._update_screen()

    def reset(
        self, seed: Optional[int] = None, options: Dict[Any, Any] = None
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed, options=options)
        self._clear_env()
        self._spawn_all()
        self._reset_rendering()
        return self.get_observation(), self._info()

    def _step_env(self, action: np.ndarray) -> bool:
        """
        Update env state from action
        :param action: player action
        :return: whether the player is stuck
        """
        self.t += 1
        self.player.velocity = self._get_agent_velocity(action)
        self._move_player()
        # resolve collisions with walls
        return not self._resolve_collisions()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        stuck = self._step_env(action)

        reward = self._reward()
        done = self._done() or stuck
        truncated = self._truncated()
        info = self._info()

        self._update_screen()

        return self.get_observation(), reward, done, truncated, info

    def _truncated(self) -> bool:
        return self.t >= self.max_steps

    def _info(self) -> dict:
        return {}

    def _get_current_image(self, observation_processing: bool = False) -> np.ndarray:
        img = pygame.surfarray.array3d(self.screen)
        exogenous_noise_config = self.observation_config.exogenous_noise_config
        if observation_processing:
            # resize image with bicubic interpolation
            img = np.array(
                Image.fromarray(img, "RGB").resize(
                    self.observation_config.img_obs_size, resample=3
                ),
                dtype=np.uint8,
            )
            if exogenous_noise_config.add_noise:
                img, self.last_exogenous_noise = add_exogenous_noise_to_image(
                    img,
                    exogenous_noise_config,
                    self._np_random,
                    self.last_exogenous_noise,
                )

        # flip to row major which is what opencv save_video is expecting.
        return np.swapaxes(img, 0, 1)

    def _get_agent_velocity(self, action):
        # apply deadzone to action --> all actions of a smaller magnitude than the deadzone are set to 0
        # this mimics common behaviour of games with game controllers where small inputs are ignored
        action = np.where(
            np.abs(action) < self.action_config.control_deadzone, 0.0, action
        )

        if self.action_config.add_transition_noise:
            # apply noise to agent action
            noise = self._np_random.normal(
                loc=self.action_config.transition_gaussian_noise_mean,
                scale=self.action_config.transition_gaussian_noise_std,
                size=2,
            )
            action += noise
        return action

    @abstractmethod
    def _done(self) -> bool:
        pass

    @abstractmethod
    def _reward(self) -> float:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation_from_state(
        self, state: np.ndarray, last_noise: np.ndarray | None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the corresponding observation for the state. Only supported for feature vector observations

        :param state: the state to convert to an observation.
        :param last_noise: last exogenous noise added to the observation in case it is needed for the conversion.
        :return: observation as a numpy array, last exogenous noise as a numpy array
        """
        pass

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        pass

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _update_ribbons(self):
        if self.rendering_config.render_player_trace:
            assert "player_trace" in self.ribbons, "Player trace ribbon not found"
            self.ribbons["player_trace"].add_point(self.player.rect.center)

    def _update_screen(self):
        self._update_ribbons()
        self.screen.fill(BACKGROUND_COLOR)
        for ribbon in self.ribbons.values():
            ribbon.draw(self.screen)
        for obj in self.objects:
            obj.draw(self.screen)
        pygame.display.update()
        pygame.event.pump()

    def render(self) -> np.ndarray:  # type: ignore[override]
        return self._get_current_image()  # type: ignore[return-value]
