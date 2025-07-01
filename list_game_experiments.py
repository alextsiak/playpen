import os
print("CLEM_GAME_REGISTRY:", os.environ.get("CLEM_GAME_REGISTRY"))
print("CWD:", os.getcwd())

from clemcore.clemgame import GameRegistry
registry = GameRegistry()
specs = registry.get_game_specs()
print(f"Found {len(specs)} specs")
for spec in specs:
    print(spec)