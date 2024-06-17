# Betbot

Betbot for tennis match predictions based on FlashResultat statistics and ATP statistics.

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
    model.ipynb
    tennis_analysis.ipynb
    data/
        tennis_matches.py
        tennis_player_data.py
        tennis_player_ids.py
        files/
            players_data.json
            players_ids.json
            matches/
                ...
        utils/
            constants.py
    models/
        best_model_fold_1.pth
        best_model_fold_2.pth
        best_model_fold_3.pth
        best_model_fold_4.pth
        best_model_fold_5.pth
    utils/
        classes.py
        constants.py
        functions.py
other/
    project_structure.py
```

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```