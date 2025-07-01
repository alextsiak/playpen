from clemcore.clemgame import GameRegistry
registry = GameRegistry()
specs = registry.get_game_specs()
print(f"Found {len(specs)} specs")
for spec in specs:
    print(spec)
