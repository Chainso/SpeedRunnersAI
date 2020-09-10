from abc import ABC, abstractmethod
from dataclasses import dataclass
from pymem import Pymem

@dataclass
class Structure(ABC):
    """
    An abstract structure with the functionality to populate the structure
    fields.
    """
    @abstractmethod
    def update(self, memory: Pymem, structure_address: int) -> None:
        """
        Updates the structure with the new values of the structure in the
        process at the address provided.

        Args:
            memory (Pymem): The instance of pymem, attached to the game.
            structure_address (int): The address of the structure.
        """
        raise NotImplementedError