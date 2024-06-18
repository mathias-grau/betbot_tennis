# Betbot - Tennis Folder

This folder contains components related to the Betbot project focused on tennis match predictions.

## Set Up 

### Important!

**ATTENTION**: Before running any scripts or notebooks, ensure to set the `ROOT_PATH` variable in the following files to the base root of your project:

- `tennis/utils/constants.py`
- `tennis/data/utils/constants.py`

Update the `REPO_PATH` variable to reflect the directory where your project is located. This ensures that all file paths are correctly referenced and the project operates as expected.

Pour les zgegs : **Comment connaître le `ROOT_PATH` ??** 
- Ouvrir un nouveau terminal directement depuis VSC (faire `control + £`) et taper `pwd` cela donne le path qu'il faut mettre dans le `ROOT_PATH` ...

### Basics 

If missing data, run : 

- `data/tennis_matches.py`
- `data/tennis_player_ids.py`
- `data/tennis_player_data.py`


## Contents

- `model.ipynb`: Jupyter notebook containing the model implementation and analysis. Run this notebook to understand the model architecture, data preprocessing steps, and insights derived from the analysis.

- `tennis_analysis.ipynb`: Jupyter notebook for additional analysis and visualization related to tennis match data and predictions.

### Data

- `data/`: Directory containing various data files used for training and analysis.

### Models

- `models/`: Directory storing trained model files used by Betbot for predictions.

### Utilities

- `utils/`: Directory containing utility scripts and modules used throughout the project.

## Running `model.ipynb`

To understand the model used in Betbot and its implementation details, follow these steps:

1. Ensure you have Jupyter Notebook installed (`pip install notebook` if not installed).
2. Navigate to the `tennis/` directory where `model.ipynb` is located.
3. Launch Jupyter Notebook directly from root path

## Future Improvements (TODO)

- **Improve Model Stability**: Refine the model in `model.ipynb` to mitigate volatility in training and validation curves.
- **Implement Feature Engineering**: Introduce advanced feature engineering techniques to enhance predictive accuracy.
- **Explore Ensemble Methods**: Investigate ensemble learning methods to combine diverse models for improved predictions.
- **Optimize Hyperparameters**: Fine-tune model hyperparameters systematically to enhance performance and generalization.
- **Enhance Data Preprocessing**: Improve data preprocessing methods to ensure higher quality input for the model, because there are a lot of missing data in the datasets.

These tasks outline areas for future development and enhancement of Betbot's predictive capabilities.

