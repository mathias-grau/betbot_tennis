from Levenshtein import distance
import numpy as np
import json
import os 
import utils.constants as c

def find_ATP_name(name : str, ATP_players_names : list):
    # use levenstein distance to find the closest name
    min_distance = np.inf
    closest_name = None
    for atp_player in ATP_players_names:
        dist = distance(name, atp_player)
        if dist < min_distance:
            min_distance = dist
            closest_name = atp_player
    return closest_name, min_distance

def update_id_table(tennis_dataset) :
    players_ids_and_names_file = f'{c.REPO_PATH}/tennis/data/files/players_ids.json'
    with open(players_ids_and_names_file, 'r') as f:
        players_ids_and_names = json.load(f)

    atp_names_to_id = players_ids_and_names['players_names']
    atp_names = list(atp_names_to_id.keys())

    fr_to_atp_ids_dict_path = f'{c.REPO_PATH}/tennis/data/files/fr_to_atp_ids.json'
    if os.path.getsize(fr_to_atp_ids_dict_path) == 0:
        correspondance_dict = {}
    else:
        with open(fr_to_atp_ids_dict_path, 'r') as f:
            correspondance_dict = json.load(f)

    for match_id in tennis_dataset.get_matches_ids() :
        fullnameplayer1 = tennis_dataset.get_match(match_id).get_players_fullname()[0]
        if len(fullnameplayer1.split('-')) == 2:
            nameplayer1 = fullnameplayer1.split('-')[-1].capitalize() + " "  + fullnameplayer1.split('-')[0].capitalize()
            nameplayer1inverted = fullnameplayer1.split('-')[0].capitalize() + " " + fullnameplayer1.split('-')[-1].capitalize()
        elif len(fullnameplayer1.split('-')) == 3:
            nameplayer1 = fullnameplayer1.split('-')[-1].capitalize() + " "  + fullnameplayer1.split('-')[0].capitalize() + " " + fullnameplayer1.split('-')[1].capitalize()
            nameplayer1inverted = fullnameplayer1.split('-')[0].capitalize() + " " + fullnameplayer1.split('-')[1].capitalize() + " " + fullnameplayer1.split('-')[-1].capitalize()
        else:
            nameplayer1 = fullnameplayer1.split('-')[-1].capitalize() + " "  + fullnameplayer1.split('-')[0].capitalize() + " " + fullnameplayer1.split('-')[1].capitalize() + " " + fullnameplayer1.split('-')[2].capitalize()
            nameplayer1inverted = fullnameplayer1.split('-')[0].capitalize() + " " + fullnameplayer1.split('-')[1].capitalize() + " " + fullnameplayer1.split('-')[2].capitalize() + " " + fullnameplayer1.split('-')[-1].capitalize()
        idplayer1 = tennis_dataset.get_match(match_id).get_players_id()[0]
        fullnameplayer2 = tennis_dataset.get_match(match_id).get_players_fullname()[1]
        if len(fullnameplayer2.split('-')) == 2:
            nameplayer2 = fullnameplayer2.split('-')[-1].capitalize() + " "  + fullnameplayer2.split('-')[0].capitalize()
            nameplayer2inverted = fullnameplayer2.split('-')[0].capitalize() + " " + fullnameplayer2.split('-')[-1].capitalize()
        elif len(fullnameplayer2.split('-')) == 3:
            nameplayer2 = fullnameplayer2.split('-')[-1].capitalize() + " "  + fullnameplayer2.split('-')[0].capitalize() + " " + fullnameplayer2.split('-')[1].capitalize()
            nameplayer2inverted = fullnameplayer2.split('-')[0].capitalize() + " " + fullnameplayer2.split('-')[1].capitalize() + " " + fullnameplayer2.split('-')[-1].capitalize()
        else:
            nameplayer2 = fullnameplayer2.split('-')[-1].capitalize() + " "  + fullnameplayer2.split('-')[0].capitalize() + " " + fullnameplayer2.split('-')[1].capitalize() + " " + fullnameplayer2.split('-')[2].capitalize()
            nameplayer2inverted = fullnameplayer2.split('-')[0].capitalize() + " " + fullnameplayer2.split('-')[1].capitalize() + " " + fullnameplayer2.split('-')[2].capitalize() + " " + fullnameplayer2.split('-')[-1].capitalize()
        idplayer2 = tennis_dataset.get_match(match_id).get_players_id()[1]
        if idplayer1 not in correspondance_dict.keys():
            # print(nameplayer1,", ", nameplayer1inverted)
            correspondant_atp_name, dist= find_ATP_name(nameplayer1, atp_names)
            correspondant_atp_name_inverted, dist_inverted = find_ATP_name(nameplayer1inverted, atp_names)
            # print(f'{nameplayer1} : {correspondant_atp_name} : {dist}')
            # print(f'{nameplayer1inverted} : {correspondant_atp_name_inverted} : {dist_inverted}')
            if dist_inverted < dist:
                correspondance_dict[idplayer1] = atp_names_to_id[correspondant_atp_name_inverted]['playerId']
                # print(f'{idplayer1} : {atp_names_to_id[correspondant_atp_name_inverted]["playerId"]}')
                # print(f'{fullnameplayer1} : {correspondant_atp_name_inverted}')
            else : 
                correspondance_dict[idplayer1] = atp_names_to_id[correspondant_atp_name]["playerId"]
                # print(f'{idplayer1} : {atp_names_to_id[correspondant_atp_name]["playerId"]}')
                # print(f'{fullnameplayer1} : {correspondant_atp_name}')

        if idplayer2 not in correspondance_dict.keys():
            print(nameplayer2,", ", nameplayer2inverted)
            correspondant_atp_name, dist = find_ATP_name(nameplayer2, atp_names)
            correspondant_atp_name_inverted, dist_inverted = find_ATP_name(nameplayer2inverted, atp_names)
            if dist_inverted < dist:
                correspondance_dict[idplayer2] = atp_names_to_id[correspondant_atp_name_inverted]['playerId']
                # print(f'{idplayer2} : {atp_names_to_id[correspondant_atp_name_inverted]["playerId"]}')
                # print(f'{fullnameplayer2} : {correspondant_atp_name_inverted}')
            else : 
                correspondance_dict[idplayer2] = atp_names_to_id[correspondant_atp_name]['playerId']
                # print(f'{idplayer2} : {atp_names_to_id[correspondant_atp_name]["playerId"]}')
                # print(f'{fullnameplayer2} : {correspondant_atp_name}')


    with open(fr_to_atp_ids_dict_path, 'w') as f:
        json.dump(correspondance_dict, f)