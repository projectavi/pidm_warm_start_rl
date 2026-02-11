# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

import numpy as np
import pygame
from pygame.sprite import Group, groupcollide

from pidm_imitation.environment.toy_env.toy_constants import (
    AGENT_COLOR,
    INACTIVE_GOAL_COLOR,
    REACHED_GOAL_COLOR,
    ROUND_POSITION,
    TRACE_COLOR,
    UNREACHED_GOAL_COLOR,
    WALL_COLOR,
)
from pidm_imitation.environment.toy_env.utils import toy_round


class BlockSprite(pygame.sprite.Sprite):
    def __init__(
        self,
        name: str,
        size: Tuple[int, int],
        color: Tuple[int, int, int],
        position: Tuple[float, float] | np.ndarray = (0, 0),
    ):
        super().__init__()
        self.name = name

        self.color = color
        self.rect = pygame.Rect(0, 0, size[0], size[1])
        self.rect.center = tuple(toy_round(position, cast_to_int=True))
        self._position = np.array(position)
        self.previous_position = np.array(position)  # copy
        self.position = np.array(position)  # set rect position.
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        return np.array(self._position)  # must make a copy

    @position.setter
    def position(self, value: np.ndarray | Tuple[float, float]) -> None:
        self.previous_position = np.array(self._position)  # copy
        if ROUND_POSITION:
            self._position = toy_round(value)
        else:
            self._position = np.array(value)

        self.rect.center = tuple(toy_round(value, cast_to_int=True))

    def update(self, velocity_scale: Tuple[float, float]):
        self.position += np.array(velocity_scale) * self.velocity

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

    def get_boundaries(self) -> Tuple[int, int, int, int]:
        return self.rect.left, self.rect.right, self.rect.top, self.rect.bottom

    def collide_with(
        self,
        other: "BlockSprite",
        other_position: np.ndarray | None = None,
    ) -> bool:
        """Check if this entity collides with another entity. If other_position is provided, check if this entity
        would collide when moved to that position.

        :param other: The other entity to check for collision with.
        :param other_position: The position of the other entity to check for collision with (optional).
        :return: True if the entities collide, False otherwise.
        """
        if other_position is not None:
            assert (
                len(other_position) == 2
            ), f"Position should be a numpy array of length 2. Got {other_position}"
            other_rect = pygame.Rect(0, 0, other.rect.width, other.rect.height)
            other_rect.center = tuple(toy_round(other_position, cast_to_int=True))
            return self.rect.colliderect(other_rect)
        return self.rect.colliderect(other.rect)


def blocksprite_collide_with_all(
    sprite: BlockSprite, group: Group, in_position: np.ndarray | None = None
) -> List[BlockSprite] | None:
    """Check if given block sprite collides with any entity in the group. If in_position is provided, check if the
    sprite would collide with any entity in the group when moved to that position.

    :param sprite: The block sprite to check for collision with the group.
    :param group: The group of entities to check for collision with.
    :param in_position: The position of the sprite to check for collision with the group (optional).
    :return: The the list of enties it collides with, or None if there is no collision.
    """
    if in_position is not None:
        assert (
            isinstance(in_position, np.ndarray) and len(in_position) == 2
        ), f"Position should be a numpy array of length 2. Got {in_position}"
        original_position = sprite.position
        original_previous_position = sprite.previous_position
        sprite.position = in_position
        collided_with = blocksprite_collide_with_all(sprite, group)
        sprite.position = original_position
        sprite.previous_position = original_previous_position
    else:
        sprite_group: Group = Group([sprite])
        collision_dict = groupcollide(group, sprite_group, False, False)  # type: ignore
        if len(collision_dict) == 0:
            return None
        collided_with = list(collision_dict.keys())

    return collided_with


def blocksprite_collide_group_with_all(
    sprite: BlockSprite,
    sprite_group: Group,
    group: Group,
    in_position: np.ndarray | None = None,
) -> List[BlockSprite] | None:
    """Check if given block sprite collides with any entity in the group. If in_position is provided, check if the
    sprite would collide with any entity in the group when moved to that position.

    :param sprite: The block sprite to check for collision with the group.
    :param group: The group of entities to check for collision with.
    :param in_position: The position of the sprite to check for collision with the group (optional).
    :return: The the list of enties it collides with, or None if there is no collision.
    """
    if in_position is not None:
        assert (
            isinstance(in_position, np.ndarray) and len(in_position) == 2
        ), f"Position should be a numpy array of length 2. Got {in_position}"
        original_position = sprite.position
        original_previous_position = sprite.previous_position
        sprite.position = in_position
        collided_with = blocksprite_collide_group_with_all(sprite, sprite_group, group)
        sprite.position = original_position
        sprite.previous_position = original_previous_position
    else:
        collision_dict = groupcollide(group, sprite_group, False, False)  # type: ignore
        if len(collision_dict) == 0:
            return None
        collided_with = list(collision_dict.keys())

    return collided_with


class Wall(BlockSprite):
    def __init__(
        self,
        position: Tuple[float, float] | np.ndarray,
        size: Tuple[int, int],
        name="wall",
        color: Tuple[int, int, int] = WALL_COLOR,
    ):
        super().__init__(name=name, size=size, color=color, position=position)

    def sign(self, x: float) -> float:
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        return 0

    def abs_max(self, x: float, y: float) -> float:
        if abs(x) > abs(y):
            return x
        return y


class Player(BlockSprite):
    def __init__(
        self,
        position: np.ndarray | Tuple[float, float],
        size: int,
        color: Tuple[int, int, int] = AGENT_COLOR,
    ):
        super().__init__("player", size=(size, size), color=color, position=position)


class Goal(BlockSprite):
    num_instances = 0

    VALID_STATES = ["inactive", "unreached", "reached"]

    def __init__(
        self,
        position: np.ndarray | Tuple[float, float],
        size: int,
        inactive_color: Tuple[int, int, int] = INACTIVE_GOAL_COLOR,
        active_unreached_color: Tuple[int, int, int] = UNREACHED_GOAL_COLOR,
        reached_color: Tuple[int, int, int] = REACHED_GOAL_COLOR,
    ):
        name = f"goal_{Goal.num_instances}"
        Goal.num_instances += 1
        super().__init__(
            name=name, size=(size, size), color=inactive_color, position=position
        )
        self.state = "inactive"
        self.inactive_color = inactive_color
        self.unreached_color = active_unreached_color
        self.reached_color = reached_color

    @property
    def reached(self) -> bool:
        return self.state == "reached"

    @property
    def active(self) -> bool:
        return self.state == "unreached"

    def set_state(self, state: str):
        if state not in Goal.VALID_STATES:
            raise ValueError(f"Invalid goal state: {state}")
        self.state = state

    def set_active(self):
        self.set_state("unreached")

    def set_reached(self):
        self.set_state("reached")

    def get_color(self) -> Tuple[int, int, int]:
        if self.state == "inactive":
            return self.inactive_color
        elif self.state == "unreached":
            return self.unreached_color
        elif self.state == "reached":
            return self.reached_color
        raise ValueError(f"Invalid goal state: {self.state}")

    def draw(self, screen: pygame.Surface):
        self.color = self.get_color()
        super().draw(screen)


class Ribbon(pygame.sprite.Sprite):

    def __init__(
        self,
        size: int,
        color: Tuple[int, int, int] = TRACE_COLOR,
        max_length: int = -1,
    ):
        super().__init__()
        self.size = size
        self.color = color
        self.max_length = max_length
        self.points: List[Tuple[int, int]] = []

    def reset(self):
        self.points = []

    def add_point(self, point: Tuple[int, int]):
        self.points.append(point)

    def draw(self, screen: pygame.Surface):
        if len(self.points) > 1:
            points = (
                self.points
                if self.max_length == -1
                else self.points[-self.max_length :]
            )
            pygame.draw.lines(screen, self.color, False, points, self.size)
        for pt in self.points:
            pygame.draw.circle(screen, self.color, pt, self.size)
