# Playpen/Clembench Troubleshooting (as of <date>)

## What Works
- The CLI (`clem list games -v`) finds all 42 games and specs.
- Registry and clembench are set up correctly for the CLI.
- All dependencies are installed, Python 3.11 is used.

## What Doesn't Work
- Direct Python API (`GameRegistry().get_game_specs()`) finds 0 specs, even with all environment variables and paths set.
- Scripts like `select_test_set.py` that use the Python API cannot generate output.

## What Was Tried
- Checked and set absolute/relative paths in `game_registry.json`.
- Set `CLEM_GAME_REGISTRY` environment variable.
- Verified clembench structure and spec files.
- Cleared caches, checked permissions, and tried importing CLI in scripts.
- Confirmed same Python interpreter is used for CLI and scripts.

## Next Steps
- Use the CLI for all registry-dependent tasks (listing games, generating test sets, etc.).
- For automation, call the CLI as a subprocess from Python scripts.
- Consider updating or reinstalling clemcore/playpen if needed.
- File a bug with clemcore if direct Python API access is required.

## How to Reproduce
- Run `clem list games -v` to see available games (works).
- Run `python3.11 list_all_specs.py` to see that Python API finds 0 specs (does not work).

## Files Added/Changed
- `list_all_specs.py`: Script to test registry access from Python.
- `README-TROUBLESHOOTING.md`: This file.
- (List any other scripts or files you changed.)
