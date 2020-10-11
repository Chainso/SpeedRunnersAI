import numpy as np

from typing import ClassVar
from dataclasses import dataclass
from pymem import Pymem

from speedrunnersai.speedrunners.structures import Structure


@dataclass
class Player(Structure):
    """
    A SpeedRunners player.
    """
    # The offsets of each property of the player
    HORIZONTAL_POSITION_OFFSET: ClassVar[int] = 0x3C
    VERTICAL_POSITION_OFFSET: ClassVar[int] = 0x40

    HORIZONTAL_VELOCITY_OFFSET: ClassVar[int] = 0x54
    VERTICAL_VELOCITY_OFFSET: ClassVar[int] = 0x58

    # The values of the player instance
    horizontal_position: float = 0
    vertical_position: float = 0

    horizontal_velocity: float = 0
    vertical_velocity: float = 0

    def update(self, memory: Pymem, player_address: int) -> None:
        self.horizontal_position = memory.read_float(
            player_address + Player.HORIZONTAL_POSITION_OFFSET
        )
        self.vertical_position = memory.read_float(
            player_address + Player.VERTICAL_POSITION_OFFSET
        )

        self.horizontal_velocity = memory.read_float(
            player_address + Player.HORIZONTAL_VELOCITY_OFFSET
        )
        self.vertical_velocity = memory.read_float(
            player_address + Player.VERTICAL_VELOCITY_OFFSET
        )

    def speed(self) -> float:
        """
        Computes the player speed from the horizontal and vertical velocity.
        """
        return np.sqrt(
            np.power(self.horizontal_velocity, 2)
            + np.power(self.vertical_velocity, 2)
        )