from typing import List
from dataclasses import dataclass, field
from pymem import Pymem

from speedrunnersai.speedrunners.structures import Address, Structure, Player

@dataclass
class Match(Structure):
    """
    A SpeedRunners match.
    """
    # Unable to find a true match structure at the moment

    # The values of the match instance
    players: List[Player] = field(default_factory = lambda: [])
    current_time: float = 0

    def update(self, memory: Pymem, match_address: int) -> None:
        # Prelimnarily using the 1 player for all
        for player in self.players:
            player.update(memory, Address.BASE_PLAYER.value)

        self.current_time = memory.read_float(
            Address.CURRENT_TIME.value
        )