import pymem

class ProcessHandler():
    """
    Handles the SpeedRunners process, with the ability to read and write to its
    memory.
    """
    def __init__(self) -> ProcessHandler:
        """
        Creates the handler, not attached to anything.
        """
        self.pymem = pymem.Pymem()

    def attach(self, process_name: str) -> None:
        """
        Attaches to the process with the given name.

        Args:
            process (str): The name of the process to attach to.
        """
        self.pymem.open_process_from_name(process_name)

    def detach(self) -> None:
        """
        Detaches from the current attached process.
        """
        self.pymem.close_process()