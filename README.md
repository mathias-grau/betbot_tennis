# Betbot

Betbot for tennis match predictions based on `FlashResultat` statistics and `ATPtour` statistics.

## Description

Betbot is a tool designed to predict outcomes of tennis matches by analyzing data from FlashResultat and ATP statistics. The model leverages historical match data to provide insights and forecasts for future matches.

## Features

- Fetches real-time match data from FlashResultat.
- Analyzes ATP player statistics.
- Provides predictions based on historical and current data.
- User-friendly command-line interface for easy interaction.

## Organisation

```bash
LICENSE
README.md
requirements.txt
tennis/
    model.ipynb                  # understand the model
    tennis_analysis.ipynb        # understand dataa
    README.md
    model_number_games.ipynb     # model for number of games in a match
    main.py                      # for future matches predictions
    long_simulation.py           # simulation during 1 year 
    test_match.ipynb             # visualize data
    data/
        tennis_matches.py        # Fetches match data from FlashResultat
        tennis_player_data.py    # Fetches player data from ATPtour
        tennis_player_ids.py     # Fetches player IDs from ATPtour
        tennis_matches_futur.py  # Fetches future match data from FlashResultat
        files/
            players_data.json
            players_ids.json
            fr_to_atp_ids.json
            errors.txt
            matches/
                ...
            future_matches/
                ...
        utils/
            constants.py
    models/
        ...
    utils/
        classes.py
        constants.py
        functions.py
```

- `tennis_matches.py`: Fetches real-time tennis match data including scores, player matchups, and match details from FlashResultat.
- `tennis_player_data.py`: Retrieves ATP player statistics such as rankings, performance metrics, and historical data from ATPtour.
- `tennis_player_ids.py`: Manages the retrieval of ATP player IDs and associated metadata required for player data lookup and analysis.


## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Important!

**ATTENTION**: Before running any scripts or notebooks, ensure to set the `ROOT_PATH` variable in the following files to the base root of your project:

- `tennis/utils/constants.py`
- `tennis/data/utils/constants.py`

Update the `ROOT_PATH` variable to reflect the directory where your project is located. This ensures that all file paths are correctly referenced and the project operates as expected.
