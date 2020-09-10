import pymem

from speedrunnersai.speedrunners.structures import Address, Player

if __name__ == "__main__":
    player = Player()
    memory = pymem.Pymem()
    memory.open_process_from_name("SpeedRunners.exe")
    """
    print("Modules")
    for module in memory.list_modules():
        print(module)
    print()
    """
    player.update(memory, Address.BASE_PLAYER.value)
    print(player)
    print(player.speed())
    memory.close_process()