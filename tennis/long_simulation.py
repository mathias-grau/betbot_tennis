# change dir to root
import os
import sys
print(os.getcwd())
if os.getcwd().split('/')[-1] == 'betbot_tennis':
    os.chdir(os.path.join(os.getcwd(), 'tennis'))
elif os.getcwd().split('/')[-1] == 'tennis':
    pass
else:
    raise ValueError('Not in root dir')
print(os.getcwd())
from utils.classes import TennisMatchDataset
import numpy as np
import pandas as pd
import json
import os 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from tqdm.notebook import tqdm
import torch 
from torch.utils.data import random_split
import utils.constants as c
import data.utils.constants as c2
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# no warning
import warnings
warnings.filterwarnings("ignore")

MAX_PADDED = 20
# Strategy results : bet the proportion given by the kelly criterion on each match
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
YELLOW = '\033[93m'
BLUE = '\033[94m'

tournaments_2023 = [   
                    'past-french-open-2023-atp-singles',
                    'past-australian-open-2023-atp-singles',
                    'past-rome-2023-atp-singles',
                    'past-madrid-2023-atp-singles',
                    'past-monte-carlo-2023-atp-singles',
                    'past-lyon-2023-atp-singles',
                    'past-barcelona-2023-atp-singles',
                    'past-munich-2023-atp-singles',
                    'past-geneva-2023-atp-singles',
                    'past-estoril-2023-atp-singles',
                    'past-houston-2023-atp-singles',
                    'past-marrakech-2023-atp-singles', 
                    'past-doha-2023-atp-singles',
                    'past-dubai-2023-atp-singles',
                    'past-indian-wells-2023-atp-singles',
                    'past-miami-2023-atp-singles',
                    'past-acapulco-2023-atp-singles',
                    'past-rio-de-janeiro-2023-atp-singles',
                    'past-santiago-2023-atp-singles',
                    'past-auckland-2023-atp-singles',
                    'past-adelaide-2023-atp-singles',
                    'past-montpellier-2023-atp-singles',
                    'past-marseille-2023-atp-singles',
                    'past-cordoba-2023-atp-singles',
                    'past-dallas-2023-atp-singles',
                    'past-stuttgart-2023-atp-singles',
                    'past-hertogenbosch-2023-atp-singles',
                    'past-halle-2023-atp-singles',
                    'past-london-2023-atp-singles',
                    'past-mallorca-2023-atp-singles',
                    'past-eastbourne-2023-atp-singles',
                    'past-wimbledon-2023-atp-singles',
                    'past-hamburg-2023-atp-singles',
                    'past-bastad-2023-atp-singles',
                    'past-gstaad-2023-atp-singles', ###
                    'past-newport-2023-atp-singles',
                    'past-umag-2023-atp-singles',
                    'past-atlanta-2023-atp-singles',
                    'past-washington-2023-atp-singles',
                    'past-cincinnati-2023-atp-singles',
                    'past-winston-salem-2023-atp-singles',
                    'past-us-open-2023-atp-singles', ###
                    'past-chengdu-2023-atp-singles',
                    'past-zhuhai-2023-atp-singles', ###
                    'past-tokyo-2023-atp-singles',
                    'past-beijing-2023-atp-singles', ###
                    'past-shanghai-2023-atp-singles',
                    'past-stockholm-2023-atp-singles',
                    'past-antwerp-2023-atp-singles', ###
                    'past-vienna-2023-atp-singles',
                    'past-basel-2023-atp-singles',
                    'past-paris-2023-atp-singles',
                    'past-metz-2023-atp-singles',
                    'past-finals-turin-2023-atp-singles',
]
tournaments_ordered_2024 = [
                    ####
                    'past-brisbane-atp-singles',
                    'past-hong-kong-atp-singles',
                    'past-auckland-atp-singles',
                    'past-adelaide-atp-singles',
                    'past-australian-open-atp-singles',
                    'past-montpellier-atp-singles',
                    'past-marseille-atp-singles',
                    'past-cordoba-atp-singles',
                    'past-dallas-atp-singles',
                    # buenos aires
                    # rotterdam
                    # delray beach
                    'past-doha-atp-singles',
                    'past-rio-de-janeiro-atp-singles',
                    'past-santiago-atp-singles',
                    'past-acapulco-atp-singles',
                    'past-dubai-atp-singles',
                    'past-indian-wells-atp-singles',
                    'past-miami-atp-singles',
                    'past-estoril-atp-singles',
                    'past-houston-atp-singles',
                    'past-marrakech-atp-singles',
                    'past-monte-carlo-atp-singles',
                    'past-munich-atp-singles',
                    'past-bucharest-atp-singles',
                    'past-barcelona-atp-singles',
                    'past-madrid-atp-singles',
                    'past-rome-atp-singles', 
                    'past-lyon-atp-singles',
                    'past-geneva-atp-singles',
                    'past-french-open-atp-singles', 
                    'past-stuttgart-atp-singles',
                    'past-hertogenbosch-atp-singles',
                    'past-halle-atp-singles',
                    'past-london-atp-singles',
                    'past-mallorca-atp-singles',
                    'past-eastbourne-atp-singles',
                    'past-wimbledon-atp-singles',
                    ]


PATIENCE = 80 # 100
N_EPOCHS = 2000 # 2000
LEARNING_RATE = 5e-4 # 5e-4
WEIGHT_DECAY = 5e-4 # 1e-6
DROPOUT = 0.4 # 0.6
N_UNITS = 8 # 16
N_UNITS_OVERALL = 8 # 4
N_UNITS_SURFACE = 8 # 4
STEP_SIZE = 15 # 30
GAMMA = 0.9 # 0.9
TOURNAMENT_HIDDEN = 2 # 2
PLAYER_HIDDEN = 16 # 20
HIDDEN = 16 # 32


class TennisMatchPredictor(nn.Module):
    def __init__(self, input_shapes):
        super(TennisMatchPredictor, self).__init__()
        self.fctournament = nn.Linear(input_shapes[0], TOURNAMENT_HIDDEN) 
        self.fcplayer1 = nn.Linear(input_shapes[1], PLAYER_HIDDEN) 
        self.fc2player1 = nn.Linear(PLAYER_HIDDEN, PLAYER_HIDDEN)
        self.fcplayer2 = nn.Linear(input_shapes[2], PLAYER_HIDDEN) 
        self.fc2player2 = nn.Linear(PLAYER_HIDDEN, PLAYER_HIDDEN)
        self.gru_h2h_overall = nn.GRU(1, N_UNITS_OVERALL, batch_first=True, dropout=DROPOUT)
        self.gru_h2h_surface = nn.GRU(1, N_UNITS_SURFACE, batch_first=True, dropout=DROPOUT)
        self.gru_shape_overall_player1 = nn.GRU(1, N_UNITS, batch_first=True, dropout=DROPOUT)
        self.gru_shape_overall_player2 = nn.GRU(1, N_UNITS, batch_first=True, dropout=DROPOUT)
        self.fc1 = nn.Linear(TOURNAMENT_HIDDEN + PLAYER_HIDDEN + PLAYER_HIDDEN + N_UNITS_OVERALL + N_UNITS_SURFACE + N_UNITS + N_UNITS, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 1)

    def forward(self, 
                tournament_features, 
                player1_features, 
                player2_features, 
                h2h_overall, 
                h2h_surface, 
                shape_overall_player1, 
                shape_overall_player2,
                tournament_mask=None,
                player1_mask=None, 
                player2_mask=None, 
                h2h_overall_mask=None, 
                h2h_surface_mask=None, 
                shape_overall_player1_mask=None, 
                shape_overall_player2_mask=None):
        
        # x1 = F.tanh(self.fctournament(tournament_features))
        x1 = tournament_features
        x2 = F.sigmoid(self.fcplayer1(player1_features))
        x2 = F.tanh(self.fc2player1(x2))
        x3 = F.sigmoid(self.fcplayer2(player2_features))
        x3 = F.tanh(self.fc2player2(x3))

        # Reverse the sequence for GRU processing
        h2h_overall = torch.flip(h2h_overall, dims=[1]).unsqueeze(-1)
        h2h_surface = torch.flip(h2h_surface, dims=[1]).unsqueeze(-1)
        shape_overall_player1 = torch.flip(shape_overall_player1, dims=[1]).unsqueeze(-1)
        shape_overall_player2 = torch.flip(shape_overall_player2, dims=[1]).unsqueeze(-1)

        if h2h_overall_mask is not None:
            h2h_overall_mask = torch.flip(h2h_overall_mask, dims=[1]).unsqueeze(-1)
        if h2h_surface_mask is not None:
            h2h_surface_mask = torch.flip(h2h_surface_mask, dims=[1]).unsqueeze(-1)
        if shape_overall_player1_mask is not None:
            shape_overall_player1_mask = torch.flip(shape_overall_player1_mask, dims=[1]).unsqueeze(-1)
        if shape_overall_player2_mask is not None:
            shape_overall_player2_mask = torch.flip(shape_overall_player2_mask, dims=[1]).unsqueeze(-1)

        def apply_gru_with_mask(gru, x, mask):
            batch_size, seq_len, _ = x.size()
            hidden = torch.zeros(batch_size, gru.hidden_size).to(x.device)
            for t in range(seq_len):
                input_t = x[:, t, :]
                mask_t = mask[:, t, :].float()
                out, hidden = gru(input_t.unsqueeze(1), hidden.unsqueeze(0))
                hidden = hidden.squeeze(0) * mask_t + hidden.squeeze(0) * (1 - mask_t)
            return hidden

        x4 = apply_gru_with_mask(self.gru_h2h_overall, h2h_overall, h2h_overall_mask)
        x4 = F.tanh(x4)

        x5 = apply_gru_with_mask(self.gru_h2h_surface, h2h_surface, h2h_surface_mask)
        x5 = F.tanh(x5)

        x6 = apply_gru_with_mask(self.gru_shape_overall_player1, shape_overall_player1, shape_overall_player1_mask)
        x6 = F.tanh(x6)

        x7 = apply_gru_with_mask(self.gru_shape_overall_player2, shape_overall_player2, shape_overall_player2_mask)
        x7 = F.tanh(x7)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), 1)
        x = F.dropout(x, p=DROPOUT)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
    
PROB_THRESHOLD = 0.5

# Kelly criterion
def kelly_criterion(odds, prob, safe=0.1):
    return (prob - ((1- prob)/(odds)))*safe



INIT_AMOUNT = 100 # euros
account_values = []
total_amount = INIT_AMOUNT
account_values.append(total_amount)

for i in tqdm(range(len(tournaments_ordered_2024))):
    print("\n\n\n")
    print(15*'-')
    print(f'{BLUE}Tournament : {tournaments_ordered_2024[i]}{RESET}')
    if i < 15 :
        print(f'{RED}Skipping tournament{RESET}')
        continue
    train_tournaments = tournaments_2023 + tournaments_ordered_2024[:i]
    test_tournamets = [tournaments_ordered_2024[i]]
    print(f'    - Train tournaments until : {train_tournaments[-1]}')
    print(f'    - Test tournament : {test_tournamets[0]}')
    tennis_dataset = TennisMatchDataset(train_tournaments, verbose = False)
    list_vectors, list_labels, lst_match_id, nb_errors = tennis_dataset.get_past_vectors(verbose = False)
    input_shapes = []
    for i in range(len(list_vectors[0])):
        input_shapes.append(len(list_vectors[0][i]))
    lst_tournaments = []
    new_list_vectors = []
    new_list_labels = []
    for i in tqdm(range(len(list_vectors))) :
        vector = list_vectors[i]
        num_padding = 0
        for spe_vec in vector :
            num_padding += spe_vec.count(c.PADDING)
        if num_padding < MAX_PADDED and list_labels[i] != c.PADDING:
            new_list_vectors.append(vector)
            new_list_labels.append(list_labels[i])
    tqdm.write(f'{YELLOW}Number of vectors after removing vectors with too much missing values : {len(new_list_vectors)} over {len(list_vectors)}{RESET}')

    if len(new_list_vectors) == 0:
        print(f'{RED}No data to train{RESET}')
        continue





    tennis_test_dataset = TennisMatchDataset(test_tournamets, verbose = False)
    nb_errors = 0
    list_vectors_test = []
    list_labels_test = []
    list_matches_ids_test = []

    list_vectors_test, list_labels_test, list_matches_ids_test, nb_errors = tennis_test_dataset.get_past_vectors(verbose = False)
    input_shapes = []
    for i in range(len(list_vectors_test[0])):
        input_shapes.append(len(list_vectors_test[0][i]))
         
    new_list_vectors_test = []
    new_list_labels_test = []
    new_list_matches_ids_test = []

    for i in tqdm(range(len(list_vectors_test))) :
        vector = list_vectors_test[i]
        num_padding = 0
        for spe_vec in vector :
            num_padding += spe_vec.count(c.PADDING)
        if num_padding < MAX_PADDED and list_labels_test[i] != c.PADDING:
            new_list_vectors_test.append(vector)
            new_list_labels_test.append(list_labels_test[i])
            new_list_matches_ids_test.append(list_matches_ids_test[i])

    if len(new_list_vectors_test) == 0:
        print(f'{RED}No data to predict{RESET}')
        continue

    
    
    
    
    
    # create 7 tensors : [tournament_features_vector, player1_features_vector, player2_features_vector, h2h_overall_vector, h2h_surface_vector, shape_overall_player1, shape_overall_player2]
    tournament_features_vector = []
    player1_features_vector = []
    player2_features_vector = []
    h2h_overall_vector = []
    h2h_surface_vector = []
    shape_overall_player1_vector = []
    shape_overall_player2_vector = []

    for vector in new_list_vectors:
        tournament_features_vector.append(vector[0])
        player1_features_vector.append(vector[1])
        player2_features_vector.append(vector[2])
        h2h_overall_vector.append(vector[3])
        h2h_surface_vector.append(vector[4])
        shape_overall_player1_vector.append(vector[5])
        shape_overall_player2_vector.append(vector[6])

    # convert to pytorch tensor
    tournament_features_tensor = torch.tensor(tournament_features_vector, dtype=torch.float)
    tournament_features_mask = torch.zeros_like(tournament_features_tensor)
    tournament_features_mask[tournament_features_tensor != c.PADDING] = 1.
    player1_features_tensor = torch.tensor(player1_features_vector, dtype=torch.float)
    player1_features_mask = torch.zeros_like(player1_features_tensor)
    player1_features_mask[player1_features_tensor != c.PADDING] = 1.
    player2_features_tensor = torch.tensor(player2_features_vector, dtype=torch.float)
    player2_features_mask = torch.zeros_like(player2_features_tensor)
    player2_features_mask[player2_features_tensor != c.PADDING] = 1.
    h2h_overall_tensor = torch.tensor(h2h_overall_vector, dtype=torch.float)
    h2h_overall_mask = torch.zeros_like(h2h_overall_tensor)
    h2h_overall_mask[h2h_overall_tensor != c.PADDING] = 1.
    h2h_surface_tensor = torch.tensor(h2h_surface_vector, dtype=torch.float)
    h2h_surface_mask = torch.zeros_like(h2h_surface_tensor)
    h2h_surface_mask[h2h_surface_tensor != c.PADDING] = 1.
    shape_overall_player1_tensor = torch.tensor(shape_overall_player1_vector, dtype=torch.float)
    shape_overall_player1_mask = torch.zeros_like(shape_overall_player1_tensor)
    shape_overall_player1_mask[shape_overall_player1_tensor != c.PADDING] = 1.
    shape_overall_player2_tensor = torch.tensor(shape_overall_player2_vector, dtype=torch.float)
    shape_overall_player2_mask = torch.zeros_like(shape_overall_player2_tensor)
    shape_overall_player2_mask[shape_overall_player2_tensor != c.PADDING] = 1.

    label_vector = []
    for label in new_list_labels:
        label_vector.append(label)

    label_tensor = torch.tensor(label_vector)

    dataset = TensorDataset(tournament_features_tensor, 
                            tournament_features_mask, 
                            player1_features_tensor, 
                            player1_features_mask, 
                            player2_features_tensor, 
                            player2_features_mask,
                            h2h_overall_tensor, 
                            h2h_overall_mask, 
                            h2h_surface_tensor, 
                            h2h_surface_mask, 
                            shape_overall_player1_tensor, 
                            shape_overall_player1_mask, 
                            shape_overall_player2_tensor, 
                            shape_overall_player2_mask, 
                            label_tensor)

    # split the dataset into train and validation 
    # train_size = int(0.85 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True)

    # Initialize lists to store the results
    all_train_losses = []
    all_val_losses = []
    all_last_indexes = []

    badly_trained_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        tqdm.write(f"Fold {fold + 1}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=64, shuffle=False)

        model = TennisMatchPredictor(input_shapes)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        fold_train_losses = []
        fold_val_losses = []

        patience_counter = 0
        MIN_VAL_LOSS = np.inf
        BEST_MODEL = None
        INDEX_EPOCH = 0
        
        for epoch in tqdm(range(N_EPOCHS)):
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            for data in train_dataloader:
                (tournament_features, tournament_mask, player1_features, player1_mask,
                player2_features, player2_mask, h2h_overall, h2h_overall_mask, 
                h2h_surface, h2h_surface_mask, shape_overall_player1, shape_overall_player1_mask, 
                shape_overall_player2, shape_overall_player2_mask, labels) = data
                optimizer.zero_grad()
                outputs = model(tournament_features, player1_features, player2_features, 
                                h2h_overall, h2h_surface, shape_overall_player1, shape_overall_player2,
                                tournament_mask, player1_mask, player2_mask, h2h_overall_mask, 
                                h2h_surface_mask, shape_overall_player1_mask, shape_overall_player2_mask)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            lr_scheduler.step()

            

            
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    (tournament_features, tournament_mask, player1_features, player1_mask,
                    player2_features, player2_mask, h2h_overall, h2h_overall_mask, 
                    h2h_surface, h2h_surface_mask, shape_overall_player1, shape_overall_player1_mask, 
                    shape_overall_player2, shape_overall_player2_mask, labels) = data
                    outputs = model(tournament_features, player1_features, player2_features, 
                                    h2h_overall, h2h_surface, shape_overall_player1, shape_overall_player2,
                                    tournament_mask, player1_mask, player2_mask, h2h_overall_mask, 
                                    h2h_surface_mask, shape_overall_player1_mask, shape_overall_player2_mask)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    val_loss += loss.item()
                

                fold_train_losses.append(train_loss / len(train_dataloader))
                fold_val_losses.append(val_loss / len(val_dataloader))

                if epoch % 100 == 0:
                    all_weights = torch.cat([x.view(-1) for x in model.parameters()])
                    tqdm.write(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataloader):.2f}, Validation Loss: {val_loss / len(val_dataloader):.2f}, lr: {lr_scheduler.get_last_lr()[0]:.2e}, Weight norm: {all_weights.norm():.2f}')
                if val_loss < MIN_VAL_LOSS:
                    MIN_VAL_LOSS = val_loss
                    patience_counter = 0
                    BEST_MODEL = model.state_dict()
                    INDEX_EPOCH = epoch
                else:
                    patience_counter += 1
                if patience_counter == PATIENCE:
                    tqdm.write(f'{YELLOW}       --> Early stopping at epoch {epoch + 1} with validation loss: {MIN_VAL_LOSS/len(val_dataloader):.2f}{RESET}')
                    break
            
            all_train_losses.append(fold_train_losses)
            all_val_losses.append(fold_val_losses)
            all_last_indexes.append(INDEX_EPOCH)
            
            # Save the best model for each fold
            torch.save(BEST_MODEL, f'{c2.REPO_PATH}/tennis/models/best_model_fold_{fold + 1}.pth')
        if MIN_VAL_LOSS/len(val_dataloader) > 0.88 : 
            badly_trained_folds.append(fold + 1)


    print(f'{YELLOW}Folds to ignore : {badly_trained_folds}{RESET}')

    tournament_features_vector_test = []
    player1_features_vector_test = []
    player2_features_vector_test = []
    h2h_overall_vector_test = []
    h2h_surface_vector_test = []
    shape_overall_player1_vector_test = []
    shape_overall_player2_vector_test = []

    for vector in new_list_vectors_test:
        tournament_features_vector_test.append(vector[0])
        player1_features_vector_test.append(vector[1])
        player2_features_vector_test.append(vector[2])
        h2h_overall_vector_test.append(vector[3])
        h2h_surface_vector_test.append(vector[4])
        shape_overall_player1_vector_test.append(vector[5])
        shape_overall_player2_vector_test.append(vector[6])

    # convert to pytorch tensor
    tournament_features_tensor_test = torch.tensor(tournament_features_vector_test, dtype=torch.float)
    tournament_features_mask_test = torch.zeros_like(tournament_features_tensor_test)
    tournament_features_mask_test[tournament_features_tensor_test != c.PADDING] = 1.
    player1_features_tensor_test = torch.tensor(player1_features_vector_test, dtype=torch.float)
    player1_features_mask_test = torch.zeros_like(player1_features_tensor_test)
    player1_features_mask_test[player1_features_tensor_test != c.PADDING] = 1.
    player2_features_tensor_test = torch.tensor(player2_features_vector_test, dtype=torch.float)
    player2_features_mask_test = torch.zeros_like(player2_features_tensor_test)
    player2_features_mask_test[player2_features_tensor_test != c.PADDING] = 1.
    h2h_overall_tensor_test = torch.tensor(h2h_overall_vector_test, dtype=torch.float)
    h2h_overall_mask_test = torch.zeros_like(h2h_overall_tensor_test)
    h2h_overall_mask_test[h2h_overall_tensor_test != c.PADDING] = 1.
    h2h_surface_tensor_test = torch.tensor(h2h_surface_vector_test, dtype=torch.float)
    h2h_surface_mask_test = torch.zeros_like(h2h_surface_tensor_test)
    h2h_surface_mask_test[h2h_surface_tensor_test != c.PADDING] = 1.
    shape_overall_player1_tensor_test = torch.tensor(shape_overall_player1_vector_test, dtype=torch.float)
    shape_overall_player1_mask_test = torch.zeros_like(shape_overall_player1_tensor_test)
    shape_overall_player1_mask_test[shape_overall_player1_tensor_test != c.PADDING] = 1.
    shape_overall_player2_tensor_test = torch.tensor(shape_overall_player2_vector_test, dtype=torch.float)
    shape_overall_player2_mask_test = torch.zeros_like(shape_overall_player2_tensor_test)
    shape_overall_player2_mask_test[shape_overall_player2_tensor_test != c.PADDING] = 1.

    label_vector = []
    for label in new_list_labels_test:
        label_vector.append(label)

    label_tensor_test = torch.tensor(label_vector)

    test_dataset = TensorDataset(tournament_features_tensor_test, 
                                tournament_features_mask_test, 
                                player1_features_tensor_test, 
                                player1_features_mask_test, 
                                player2_features_tensor_test, 
                                player2_features_mask_test,
                                h2h_overall_tensor_test, 
                                h2h_overall_mask_test, 
                                h2h_surface_tensor_test, 
                                h2h_surface_mask_test, 
                                shape_overall_player1_tensor_test, 
                                shape_overall_player1_mask_test, 
                                shape_overall_player2_tensor_test, 
                                shape_overall_player2_mask_test, 
                                label_tensor_test)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_predictions = []
    for j in range(N_FOLDS):
        if j + 1 not in badly_trained_folds:
            model = TennisMatchPredictor(input_shapes)
            model.load_state_dict(torch.load(f'{c2.REPO_PATH}/tennis/models/best_model_fold_{j + 1}.pth'))
            model.eval()
            predictions = []
            with torch.no_grad():
                test_loss = 0.0
                for i, data in enumerate(test_dataloader):
                    tournament_features, tournament_mask, player1_features, player1_mask, player2_features, player2_mask, h2h_overall, h2h_overall_mask, h2h_surface, h2h_surface_mask, shape_overall_player1, shape_overall_player1_mask, shape_overall_player2, shape_overall_player2_mask, labels = data
                    outputs = model(tournament_features = tournament_features, 
                                    player1_features = player1_features, 
                                    player2_features = player2_features, 
                                    h2h_overall = h2h_overall, 
                                    h2h_surface = h2h_surface, 
                                    shape_overall_player1 = shape_overall_player1, 
                                    shape_overall_player2 = shape_overall_player2,
                                    tournament_mask = tournament_mask, 
                                    player1_mask = player1_mask, 
                                    player2_mask = player2_mask, 
                                    h2h_overall_mask = h2h_overall_mask,
                                    h2h_surface_mask = h2h_surface_mask,
                                    shape_overall_player1_mask = shape_overall_player1_mask,
                                    shape_overall_player2_mask = shape_overall_player2_mask
                                    )    
                    predictions.append(outputs)        
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    test_loss += loss.item()
                print(f'Test Loss: for {j+1} : {test_loss/len(test_dataloader):.2f}')
            # find the matches where the model is the most confident and was right
            predictions = torch.cat(predictions).flatten()
            all_predictions.append(predictions)

    all_predictions = torch.stack(all_predictions)
    predictions = all_predictions.mean(dim=0)

    print(f'{YELLOW}Making predictions for {len(predictions)} matches for tournament {test_tournamets[0]}{RESET}')

    # get the indexes of the matches where the model was right
    # create data frame with the predictions and the labels and the match ids
    df = pd.DataFrame(columns=['match_id', 'predictions', 'labels'])
    df['predictions'] = predictions
    df['labels'] = label_tensor_test
    df['match_id'] = new_list_matches_ids_test
    # order the data frame by predictions values
    df = df.sort_values(by='predictions', ascending=False)

    # fetch the odds of betclic for the matches
    odds_1_list = []
    odds_2_list = []
    prob_list = []
    for match_id in df['match_id']:
        match_odds =tennis_test_dataset.get_match(match_id).get_odds()
        odds_found = False
        for match_odd in match_odds:
            if match_odd['bookmaker'] == 'Betclic.fr':
                odds_found = True
                odds_1_list.append(float(match_odd['odds'][0]))
                odds_2_list.append(float(match_odd['odds'][1]))
                player1odd = float(match_odd['odds'][0])
                player2odd = float(match_odd['odds'][1])
                prob_win_player1 = 1/player1odd
                prob_win_player2 = 1/player2odd
                # normalize the probabilities
                prob_sum = prob_win_player1 + prob_win_player2
                prob_win_player1 = prob_win_player1/prob_sum
                prob_win_player2 = prob_win_player2/prob_sum
                prob_list.append(2*prob_win_player1-1)
                break
        if not odds_found:
            odds_1_list.append(None)
            odds_2_list.append(None)
            prob_list.append(None) 

    df['odds_1'] = odds_1_list
    df['odds_2'] = odds_2_list
    df['bookmaker_pred'] = prob_list
    df = df.loc[df['odds_1'].notnull()]

    # if predictions > 0.6 bet on player 1, if predictions < -0.6 bet on player 2
    bet_on_player_1_df = df.loc[df['predictions'] > PROB_THRESHOLD].copy()
    bet_on_player_1_df['prediction_prob']= bet_on_player_1_df['predictions'].apply(lambda x : (1+x)/2)
    bet_on_player_1_df['kelly_criterion'] = bet_on_player_1_df.apply(lambda row : kelly_criterion(row['odds_1'], row['prediction_prob']), axis=1)

    bet_on_player_2_df = df.loc[df['predictions'] < -PROB_THRESHOLD].copy()
    bet_on_player_2_df['prediction_prob']= bet_on_player_2_df['predictions'].apply(lambda x : (1-x)/2)
    bet_on_player_2_df['kelly_criterion'] = bet_on_player_2_df.apply(lambda row : kelly_criterion(row['odds_2'], row['prediction_prob']), axis=1)
    
    if len(bet_on_player_1_df) == 0 and len(bet_on_player_2_df) == 0:
        print(f'{RED}No bets to make{RESET}')
        continue

    for i, row in bet_on_player_1_df.iterrows():
        player1 = tennis_test_dataset.get_match(row['match_id']).get_players()[0]
        player2 = tennis_test_dataset.get_match(row['match_id']).get_players()[1]
        amout_to_bet = max(row['kelly_criterion']*total_amount, 0)
        if row['labels'] == 1:
            total_amount += amout_to_bet*(row['odds_1']-1)
            print(f"{GREEN}Match : {row['match_id']} {player1} - {player2}, bet on player 1 ({row['prediction_prob']:.2f}), amount to bet : {amout_to_bet:.2f}, odds : {row['odds_1']:.2f}, label : {row['labels']:.2f}, total amount : {total_amount:.2f}{RESET}")
        else:
            total_amount -= amout_to_bet
            print(f"{RED}Match : {row['match_id']} {player1} - {player2}, bet on player 1 ({row['prediction_prob']:.2f}), amount to bet : {amout_to_bet:.2f}, odds : {row['odds_1']:.2f}, label : {row['labels']:.2f}, total amount : {total_amount:.2f}{RESET}")
        account_values.append(total_amount)

    for i, row in bet_on_player_2_df.iterrows():
        player1 = tennis_test_dataset.get_match(row['match_id']).get_players()[0]
        player2 = tennis_test_dataset.get_match(row['match_id']).get_players()[1]
        amout_to_bet = max(row['kelly_criterion']*total_amount, 0)
        if row['labels'] == -1:
            total_amount += amout_to_bet*(row['odds_2']-1)
            print(f"{GREEN}Match : {row['match_id']} {player1} - {player2}, bet on player 2 ({row['prediction_prob']:.2f}), amount to bet : {amout_to_bet:.2f}, odds : {row['odds_2']:.2f}, label : {row['labels']:.2f}, total amount : {total_amount:.2f}{RESET}")
        else:
            total_amount -= amout_to_bet
            print(f"{RED}Match : {row['match_id']} {player1} - {player2}, bet on player 2 ({row['prediction_prob']:.2f}), amount to bet : {amout_to_bet:.2f}, odds : {row['odds_2']:.2f}, label : {row['labels']:.2f}, total amount : {total_amount:.2f}{RESET}")
        account_values.append(total_amount)
    

print(f"\nTotal amount after betting : {total_amount:.2f}")
if total_amount >= INIT_AMOUNT:
    print(f"{GREEN}Relative won {(total_amount-INIT_AMOUNT)/INIT_AMOUNT*100:.2f}%{RESET}")
else:
    print(f"{RED}Relative lost {(INIT_AMOUNT-total_amount)/INIT_AMOUNT*100:.2f}%{RESET}")

plt.figure(figsize=(20, 5))
plt.plot(account_values, label='Account value', color='blue', marker='o')
plt.xlabel('Match')
plt.ylabel('Amount')
plt.title('Account value over time')
plt.grid()
plt.tight_layout()
plt.show()