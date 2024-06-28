from torch.utils.data import Dataset
import utils.constants as c
from utils.functions import update_id_table
import os 
import json
import time
from tqdm import tqdm


def find_name_in_dict(name, players_ids_and_names):
    # use levenstein distance to find the closest name
    for player in players_ids_and_names["players_names"].keys():
        if name in player:
            return player
        else :
            if player.split(" ")[0] in name:
                return player
    

class TennisMatch:
    def __init__(self, match_id: str, match_data: dict): 
        self.match_id = match_id
        self.tournament = match_data["tournament"]
        self.league = match_data["league"]
        # match_data["date"] is format string "09.06.2024 15:10" 
        self.date = time.mktime(time.strptime(match_data["date"], "%d.%m.%Y %H:%M"))
        
        self.player1idFR = match_data["players"]["player1"]["id"]
        self.player2idFR = match_data["players"]["player2"]["id"]
        self.player1name = match_data["players"]["player1"]["name"]
        self.player2name = match_data["players"]["player2"]["name"]
        self.player1fullname = match_data["players"]["player1"]["fullname"]
        self.player2fullname = match_data["players"]["player2"]["fullname"]
        # if match_data["result"]["status"] == "FINISHED" or match_data["result"]["status"] == "INPROGRESS" or match_data["result"]["status"].split("-")[0] == "FINISHED / RETIRED ":
        if match_data["result"] != {} :
            if match_data["result"]["status"] == "Finished" :
                self.setsplayer1 = int(match_data["result"]["player1"])
                self.setsplayer2 = int(match_data["result"]["player2"])
                self.winner = 1 if self.setsplayer1 > self.setsplayer2 else -1
            else:
                self.setsplayer1 = c.PADDING
                self.setsplayer2 = c.PADDING
                self.winner = c.PADDING
        else:
            self.setsplayer1 = c.PADDING
            self.setsplayer2 = c.PADDING
            self.winner = c.PADDING
        self.odds = match_data["odds"]

    def __str__(self):
        return f"{self.player1name} vs {self.player2name} : {self.setsplayer1} - {self.setsplayer2}"
    
    def get_players(self):
        return (self.player1name, self.player2name)
    
    def get_players_id(self):
        return (self.player1idFR, self.player2idFR)
    
    def get_players_fullname(self):
        return (self.player1fullname, self.player2fullname)
    
    def get_label(self):
        return self.winner
    
    def get_sets(self):
        return (self.setsplayer1, self.setsplayer2)
    
    def get_tournament(self):
        return self.tournament
    
    def get_match_id(self):
        return self.match_id
    
    def get_players_atp_id(self):
        with open(c.CORRESPONDANCE_FR_IDS_ATP_IDS_PATH, "r") as f:
            correspondance_frid_to_atpid = json.load(f)
        player1idATP = correspondance_frid_to_atpid[self.player1idFR]
        player2idATP = correspondance_frid_to_atpid[self.player2idFR]
        return (player1idATP, player2idATP)
    
    def get_players_atp_data(self):
        player1idATP, player2idATP = self.get_players_atp_id()
        if os.path.exists(c.PLAYERS_DATA_PATH):
            with open(c.PLAYERS_DATA_PATH, "r") as f:
                players_data = json.load(f)
            player1data = players_data[player1idATP]
            # player1rank is indice of the player in the players_data
            player1rank = list(players_data.keys()).index(player1idATP)+1
            player1data["rank"] = player1rank
            player2data = players_data[player2idATP]
            player2rank = list(players_data.keys()).index(player2idATP)+1
            player2data["rank"] = player2rank
            return (player1data, player2data)
        else:
            print("No data for players")
            return None
    
    def get_match_data(self):
        if self.date < time.time():
            path = os.path.join(c.TENNIS_DATA_PATH, f"past-{self.tournament}-{self.league}.json")
        else:
            path = os.path.join(c.TENNIS_DATA_PATH, f"{self.tournament}-{self.league}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                tournament_data = json.load(f)
            return tournament_data[self.match_id]
        else:
            print(f"No data for {self.tournament}")
            return None
    
    def get_odds(self):
        return self.odds
    
    def get_feature_vector(self):
        player1data, player2data = self.get_players_atp_data()
        match_data = self.get_match_data()
        # features about tournament 
        tournament_features_vector = []
        tournament_features_vector.append(c.MAX_TOURNAMENTS_TYPE_VALUE - float(c.TOURNAMENTS_TYPE[self.tournament])/(c.MAX_TOURNAMENTS_TYPE_VALUE - 1) if self.tournament in c.TOURNAMENTS_TYPE.keys() else c.PADDING)
        tournament_features_vector.append(float(c.SURFACE_TYPE[c.TOURNAMENTS_SURFACE[self.tournament]])/c.MAX_SURFACE_TYPE_VALUE if self.tournament in c.TOURNAMENTS_SURFACE.keys() else c.PADDING)

        def create_player_feature_vector(playerdata) : 
            player_features_vector = []
            # specificities of the player
            player_features_vector.append(float(playerdata["rank"])/1000 if playerdata["rank"] != None else c.PADDING)
            player_features_vector.append(float(playerdata["data"]["age"].split(" ")[0])/50 if playerdata["data"]["age"] != "" else c.PADDING)
            if playerdata["data"]["height"] != "(cm)":
                player_features_vector.append(float(playerdata["data"]["height"].split("(")[-1].split("cm")[0])/250 if playerdata["data"]["height"] != "" else c.PADDING)
            else:
                player_features_vector.append(c.PADDING)
            # player_features_vector.append(float(playerdata["data"]["weight"].split("(")[-1].split("kg")[0]) if playerdata["data"]["weight"] != "" else c.PADDING)
            player_features_vector.append(float(c.TYPE_PLAY[playerdata["data"]["typePlays"]])/len(c.TYPE_PLAY) if playerdata["data"]["typePlays"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Break Points Saved"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Service Games Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Total Service Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Break Points Converted"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Games Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Total Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["1st Serve"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Break Points Saved"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Service Games Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["serve"]["Total Service Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
            # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Break Points Converted"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Return Games Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Return Points Won"] != "" else c.PADDING)
            player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[self.tournament]]["return"]["Total Points Won"] != "" else c.PADDING)
            if playerdata["match_records"] == {}:
                player_features_vector.extend([c.PADDING]*22)
            else :
                player_features_vector.append(float(playerdata["match_records"]["Overall"]["ytd_index"]) if playerdata["match_records"]["Overall"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Overall"]["career_index"]) if playerdata["match_records"]["Overall"]["career_index"] != "" else c.PADDING)
                # player_features_vector.append(float(playerdata["match_records"]["Overall"]["career_titles"]) if playerdata["match_records"]["Overall"]["career_titles"] != "" else c.PADDING)
                # TODO same for "Grand Slams", "ATP Masters 1000"
                player_features_vector.append(float(playerdata["match_records"]["Tie breaks"]["ytd_index"]) if playerdata["match_records"]["Tie breaks"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Tie breaks"]["career_index"]) if playerdata["match_records"]["Tie breaks"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Versus Top 10"]["ytd_index"]) if playerdata["match_records"]["Versus Top 10"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Versus Top 10"]["career_index"]) if playerdata["match_records"]["Versus Top 10"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Finals"]["ytd_index"]) if playerdata["match_records"]["Finals"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Finals"]["career_index"]) if playerdata["match_records"]["Finals"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Deciding set"]["ytd_index"]) if playerdata["match_records"]["Deciding set"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["Deciding set"]["career_index"]) if playerdata["match_records"]["Deciding set"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["5th Set Record"]["ytd_index"]) if playerdata["match_records"]["5th Set Record"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["5th Set Record"]["career_index"]) if playerdata["match_records"]["5th Set Record"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"][c.TOURNAMENTS_SURFACE[self.tournament].capitalize()]["ytd_index"]) if playerdata["match_records"][c.TOURNAMENTS_SURFACE[self.tournament].capitalize()]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"][c.TOURNAMENTS_SURFACE[self.tournament].capitalize()]["career_index"]) if playerdata["match_records"][c.TOURNAMENTS_SURFACE[self.tournament].capitalize()]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["After Winning 1st Set"]["ytd_index"]) if playerdata["match_records"]["After Winning 1st Set"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["After Winning 1st Set"]["career_index"]) if playerdata["match_records"]["After Winning 1st Set"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["After Losing 1st Set"]["ytd_index"]) if playerdata["match_records"]["After Losing 1st Set"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["After Losing 1st Set"]["career_index"]) if playerdata["match_records"]["After Losing 1st Set"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["vs. Right Handers*"]["ytd_index"]) if playerdata["match_records"]["vs. Right Handers*"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["vs. Right Handers*"]["career_index"]) if playerdata["match_records"]["vs. Right Handers*"]["career_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["vs. Left Handers*"]["ytd_index"]) if playerdata["match_records"]["vs. Left Handers*"]["ytd_index"] != "" else c.PADDING)
                player_features_vector.append(float(playerdata["match_records"]["vs. Left Handers*"]["career_index"]) if playerdata["match_records"]["vs. Left Handers*"]["career_index"] != "" else c.PADDING)
            year = time.localtime(self.date).tm_year
            shape_player = []
            for name, tournament in playerdata['results'][str(year)].items():
                date_string = tournament['date']
                date_tuple = time.strptime(date_string, "%d %b, %y")
                date_number = time.mktime(date_tuple)
                surface = tournament['surface']
                if c.TOURNAMENTS_TYPE[self.tournament] == 1 :
                    time_lapse = 14 * 60*60*24 # 14 days 
                else :
                    time_lapse = 7 * 60*60*24 # 7 days
                if date_number < self.date - time_lapse :
                    matches = tournament['matches']
                    for match in matches :
                        opponent_name = match['opponent_name']
                        opponent_rank = match['opponent_rank']
                        result = match['result']
                        shape_player.append(result)
            while len(shape_player) < c.MAX_MATCHES_FORM:
                shape_player.append(c.PADDING)
            return player_features_vector, shape_player[:c.MAX_MATCHES_FORM]
                        

        # feature about player 1
        player1_features_vector, shape_overall_player1 = create_player_feature_vector(player1data)
        # feature about player 2
        player2_features_vector, shape_overall_player2 = create_player_feature_vector(player2data)

        h2h_overall_vector = []
        h2h_surface_vector = []
        # match statistics
        h2h_overall = match_data["h2h_overall"][2]["data"]
        h2h_surface = match_data[f"h2h_{c.TOURNAMENTS_SURFACE[self.tournament]}"][2]["data"]
        if h2h_overall == [] :
            h2h_overall_vector =  4*[c.PADDING]
        else:
            i=0
            for previous_match in h2h_overall:
                match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
                if match_date < self.date - 60*60*24:
                    i+=1
                    if previous_match["resultPlayer1"] == "-":
                        continue
                    specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
                    if previous_match[specific_match_winner] == self.player1name:
                        h2h_overall_vector.append(1)
                    else:
                        h2h_overall_vector.append(-1)
                if i == 4:
                    break
            # pad with 0 if not enough matches
            while len(h2h_overall_vector) < 4:
                h2h_overall_vector.append(c.PADDING)
        if h2h_surface == [] :
            h2h_surface_vector = 4*[c.PADDING]
        else:
            i=0
            for previous_match in h2h_surface: 
                match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
                if match_date < self.date - 60*60*24:
                    i+=1
                    if previous_match["resultPlayer1"] == "-":
                        continue
                    specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
                    if previous_match[specific_match_winner] == self.player1name:
                        h2h_surface_vector.append(1)
                    else:
                        h2h_surface_vector.append(-1)
                if i == 4:
                    break
            # pad with 0 if not enough matches
            while len(h2h_surface_vector) < 4:
                h2h_surface_vector.append(c.PADDING)
        # shape player 1 
        # shape_overall_player1 = []
        # shape_overall_player2 = []
        # last_matches_overall_player1 = match_data["h2h_overall"][0]["data"]
        # i = 0
        # for previous_match in last_matches_overall_player1:
        #     match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
        #     if match_date < self.date - 60*60*24:
        #         i+=1
        #         if previous_match["resultPlayer1"] == "-":
        #             continue
        #         specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
        #         if previous_match[specific_match_winner] == self.player1name:
        #             shape_overall_player1.append(1)
        #         else:
        #             shape_overall_player1.append(-1)
        #     if i == 4:
        #         break
        # # pad with 0 if not enough matches
        # while len(shape_overall_player1) < 4:
        #     shape_overall_player1.append(0)
        # last_matches_overall_player2 = match_data["h2h_overall"][1]["data"]
        # i = 0
        # for previous_match in last_matches_overall_player2:
        #     match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
        #     if match_date < self.date - 60*60*24:
        #         i+=1
        #         if previous_match["resultPlayer1"] == "-":
        #             continue
        #         specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
        #         if previous_match[specific_match_winner] == self.player1name:
        #             shape_overall_player2.append(1)
        #         else:
        #             shape_overall_player2.append(-1)
        #     if i == 4:
        #         break
        # # pad with 0 if not enough matches
        # while len(shape_overall_player2) < 4:
        #     shape_overall_player2.append(0)
        features_vector = [tournament_features_vector, 
                           player1_features_vector, 
                           player2_features_vector, 
                           h2h_overall_vector, 
                           h2h_surface_vector, 
                           shape_overall_player1, 
                           shape_overall_player2
                           ]
        return features_vector

class TennisMatchDataset(Dataset):
    def __init__(self, tournaments : list):
        self.tournaments = set(tournaments)
        print(f"Loading data for tournaments {self.tournaments} ...")
        self.matches = {}
        for tournament in tournaments:
            # read .json file
            tournament_path = os.path.join(c.TENNIS_DATA_PATH, f"{tournament}.json")
            with open(tournament_path, "r") as f:
                tournament_data = json.load(f)
            # update matches to self.matches
            for match_id, match_data in tournament_data.items():
                self.matches[match_id] = TennisMatch(match_id, match_data)
        print(f"... loaded {len(self.matches)} matches")
        print(f"Updating id table ...")
        update_id_table(self)
        print(f"... id table updated")
    
    def __len__(self):
        return len(list(self.matches.keys()))
    
    def __str__(self):
        return f"TennisMatchDataset with {len(self.matches)} matches and tournaments {self.tournaments}"
    
    def get_match(self, match_id):
        return self.matches[match_id]
    
    def get_matches_ids(self):
        return list(self.matches.keys())
    
    def get_match_data(self,tournament_data, match_id):
        return tournament_data[match_id]
    
    def get_player_atp_id(self, correspondance_frid_to_atpid, playeridFR):
        return correspondance_frid_to_atpid[playeridFR]
    
    def get_players_atp_data(self, correspondance_frid_to_atpid, players_data, playeridFR):
        playeridATP = self.get_player_atp_id(correspondance_frid_to_atpid, playeridFR)
        playerdata = players_data[playeridATP]
        # player1rank is indice of the player in the players_data
        playerrank = list(players_data.keys()).index(playeridATP)+1
        playerdata["rank"] = playerrank
        return playerdata
        
    
    def get_past_vectors(self):
        if os.path.exists(c.PLAYERS_DATA_PATH):
            with open(c.PLAYERS_DATA_PATH, "r") as f:
                players_data = json.load(f)
        else : 
            tqdm.write(f"Cannot find {c.PLAYERS_DATA_PATH}")
        if os.path.exists(c.CORRESPONDANCE_FR_IDS_ATP_IDS_PATH):
            with open(c.CORRESPONDANCE_FR_IDS_ATP_IDS_PATH, "r") as f:
                    correspondance_frid_to_atpid = json.load(f)
        else :
            tqdm.write(f"Cannot find {c.CORRESPONDANCE_FR_IDS_ATP_IDS_PATH}")
        features_vectors = []
        label_vector = []
        lst_match_id = []
        num_errors = 0
        for tournament in tqdm(self.tournaments):
            tournament_name = '-'.join(tournament.split("-")[1:-2])
            path = os.path.join(c.TENNIS_DATA_PATH, f"{tournament}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    tournament_data = json.load(f)
            else:
                tqdm.write(f"No data for {tournament} ({path})")
                continue
            for match_id, match_data in tournament_data.items():
                try : 
                    true_match_date = time.mktime(time.strptime(match_data["date"], "%d.%m.%Y %H:%M"))
                    player1data = self.get_players_atp_data(correspondance_frid_to_atpid, players_data, match_data["players"]["player1"]["id"])
                    player1name = match_data["players"]["player1"]["name"]
                    player2data = self.get_players_atp_data(correspondance_frid_to_atpid, players_data, match_data["players"]["player2"]["id"])
                    tournament_features_vector = []
                    tournament_features_vector.append((c.MAX_TOURNAMENTS_TYPE_VALUE - float(c.TOURNAMENTS_TYPE[tournament_name]) )/(c.MAX_TOURNAMENTS_TYPE_VALUE -1) if tournament_name in c.TOURNAMENTS_TYPE.keys() else c.PADDING)
                    tournament_features_vector.append(float(c.SURFACE_TYPE[c.TOURNAMENTS_SURFACE[tournament_name]])/c.MAX_SURFACE_TYPE_VALUE if tournament_name in c.TOURNAMENTS_SURFACE.keys() else c.PADDING)

                    def create_player_feature_vector(playerdata) : 
                        player_features_vector = []
                        # specificities of the player
                        player_features_vector.append(float(playerdata["rank"])/1000 if playerdata["rank"] != None else c.PADDING)
                        player_features_vector.append(float(playerdata["data"]["age"].split(" ")[0])/50 if playerdata["data"]["age"] != "" else c.PADDING)
                        if playerdata["data"]["height"] != "(cm)":
                            player_features_vector.append(float(playerdata["data"]["height"].split("(")[-1].split("cm")[0])/250 if playerdata["data"]["height"] != "" else c.PADDING)
                        else:
                            player_features_vector.append(c.PADDING)
                        # player_features_vector.append(float(playerdata["data"]["weight"].split("(")[-1].split("kg")[0]) if playerdata["data"]["weight"] != "" else c.PADDING)
                        player_features_vector.append(float(c.TYPE_PLAY[playerdata["data"]["typePlays"]])/c.MAX_TYPE_PLAY_VALUE if playerdata["data"]["typePlays"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Break Points Saved"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Service Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Total Service Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Break Points Converted"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["all"]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Total Points Won"] != "" else c.PADDING)
                        # # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["1st Serve"] != "" else c.PADDING)
                        # # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
                        # # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Break Points Saved"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Service Games Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["serve"]["Total Service Points Won"] != "" else c.PADDING)
                        # # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
                        # # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Break Points Converted"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Return Games Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"][c.TOURNAMENTS_SURFACE[tournament_name]]["return"]["Total Points Won"] != "" else c.PADDING)
                        ###
                        # player_features_vector.append(float(playerdata["statistics"]["Hard"]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["Hard"]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["Hard"]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Break Points Saved"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Service Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Total Service Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Break Points Converted"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["hard"]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Total Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Break Points Saved"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Service Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Total Service Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Break Points Converted"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["clay"]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Total Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["1st Serve"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["1st Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["1st Serve Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["2nd Serve Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["2nd Serve Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["Break Points Saved"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Break Points Saved"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["Service Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Service Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["serve"]["Total Service Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["serve"]["Total Service Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["1st Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["1st Serve Return Points Won"] != "" else c.PADDING)
                        # player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["2nd Serve Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["2nd Serve Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["Break Points Converted"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Break Points Converted"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["Return Games Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Games Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["Return Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Return Points Won"] != "" else c.PADDING)
                        player_features_vector.append(float(playerdata["statistics"]["grass"]["return"]["Total Points Won"].split("%")[0])/100 if playerdata["statistics"]["all"]["return"]["Total Points Won"] != "" else c.PADDING)


                        if playerdata["match_records"] == {}:
                            player_features_vector.extend([c.PADDING]*22)
                        else :
                            player_features_vector.append(float(playerdata["match_records"]["Overall"]["ytd_index"]) if playerdata["match_records"]["Overall"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Overall"]["career_index"]) if playerdata["match_records"]["Overall"]["career_index"] != "" else c.PADDING)
                            # player_features_vector.append(float(playerdata["match_records"]["Overall"]["career_titles"]) if playerdata["match_records"]["Overall"]["career_titles"] != "" else c.PADDING)
                            # TODO same for "Grand Slams", "ATP Masters 1000"
                            player_features_vector.append(float(playerdata["match_records"]["Tie breaks"]["ytd_index"]) if playerdata["match_records"]["Tie breaks"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Tie breaks"]["career_index"]) if playerdata["match_records"]["Tie breaks"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Versus Top 10"]["ytd_index"]) if playerdata["match_records"]["Versus Top 10"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Versus Top 10"]["career_index"]) if playerdata["match_records"]["Versus Top 10"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Finals"]["ytd_index"]) if playerdata["match_records"]["Finals"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Finals"]["career_index"]) if playerdata["match_records"]["Finals"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Deciding set"]["ytd_index"]) if playerdata["match_records"]["Deciding set"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["Deciding set"]["career_index"]) if playerdata["match_records"]["Deciding set"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["5th Set Record"]["ytd_index"]) if playerdata["match_records"]["5th Set Record"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["5th Set Record"]["career_index"]) if playerdata["match_records"]["5th Set Record"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"][c.TOURNAMENTS_SURFACE[tournament_name].capitalize()]["ytd_index"]) if playerdata["match_records"][c.TOURNAMENTS_SURFACE[tournament_name].capitalize()]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"][c.TOURNAMENTS_SURFACE[tournament_name].capitalize()]["career_index"]) if playerdata["match_records"][c.TOURNAMENTS_SURFACE[tournament_name].capitalize()]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["After Winning 1st Set"]["ytd_index"]) if playerdata["match_records"]["After Winning 1st Set"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["After Winning 1st Set"]["career_index"]) if playerdata["match_records"]["After Winning 1st Set"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["After Losing 1st Set"]["ytd_index"]) if playerdata["match_records"]["After Losing 1st Set"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["After Losing 1st Set"]["career_index"]) if playerdata["match_records"]["After Losing 1st Set"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["vs. Right Handers*"]["ytd_index"]) if playerdata["match_records"]["vs. Right Handers*"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["vs. Right Handers*"]["career_index"]) if playerdata["match_records"]["vs. Right Handers*"]["career_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["vs. Left Handers*"]["ytd_index"]) if playerdata["match_records"]["vs. Left Handers*"]["ytd_index"] != "" else c.PADDING)
                            player_features_vector.append(float(playerdata["match_records"]["vs. Left Handers*"]["career_index"]) if playerdata["match_records"]["vs. Left Handers*"]["career_index"] != "" else c.PADDING)
                        year = time.localtime(true_match_date).tm_year
                        shape_player = []
                        for name, tournament in playerdata['results'][str(year)].items():
                            date_string = tournament['date']
                            date_tuple = time.strptime(date_string, "%d %b, %y")
                            date_number = time.mktime(date_tuple)
                            surface = tournament['surface']
                            if c.TOURNAMENTS_TYPE[tournament_name] == 1 :
                                time_lapse = 14 * 60*60*24 # 14 days 
                            else :
                                time_lapse = 7 * 60*60*24 # 7 days
                            if date_number < true_match_date - time_lapse :
                                matches = tournament['matches']
                                for match in matches :
                                    opponent_name = match['opponent_name']
                                    opponent_rank = match['opponent_rank']
                                    result = match['result']
                                    shape_player.append(result)
                        while len(shape_player) < c.MAX_MATCHES_FORM:
                            shape_player.append(c.PADDING)
                        return player_features_vector, shape_player[:c.MAX_MATCHES_FORM]
                    
                    # feature about player 1
                    player1_features_vector, shape_overall_player1 = create_player_feature_vector(player1data)
                    # feature about player 2
                    player2_features_vector, shape_overall_player2 = create_player_feature_vector(player2data)
                
                    
                    # match statistics
                    if match_data["h2h_overall"] == [] :
                        h2h_overall_vector = 4*[c.PADDING]
                    
                    else:
                        h2h_overall_vector = []
                        h2h_overall = match_data["h2h_overall"][2]["data"]
                        i=0
                        for previous_match in h2h_overall:
                            match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
                            if match_date < true_match_date - 60*60*24:
                                i+=1
                                if previous_match["resultPlayer1"] == "-" or previous_match["resultPlayer1"] == None:
                                    continue
                                specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
                                if previous_match[specific_match_winner] == player1name:
                                    h2h_overall_vector.append(1)
                                else:
                                    h2h_overall_vector.append(-1)
                            if i == 4:
                                break
                        # pad with 0 if not enough matches
                        while len(h2h_overall_vector) < 4:
                            h2h_overall_vector.append(c.PADDING)

                    if match_data[f"h2h_{c.TOURNAMENTS_SURFACE[tournament_name]}"] == [] :
                        h2h_surface_vector = 4*[c.PADDING]
                    else:
                        h2h_surface = match_data[f"h2h_{c.TOURNAMENTS_SURFACE[tournament_name]}"][2]["data"]
                        h2h_surface_vector = []
                        i=0
                        for previous_match in h2h_surface: 
                            match_date = time.mktime(time.strptime(previous_match["date"], "%d.%m.%y"))
                            if match_date < true_match_date - 60*60*24:
                                i+=1
                                if previous_match["resultPlayer1"] == "-" or previous_match["resultPlayer1"] == None:
                                    continue
                                specific_match_winner = "player1" if int(previous_match["resultPlayer1"]) > int(previous_match["resultPlayer2"]) else "player2"
                                if previous_match[specific_match_winner] == player1name:
                                    h2h_surface_vector.append(1)
                                else:
                                    h2h_surface_vector.append(-1)
                            if i == 4:
                                break
                        # pad with 0 if not enough matches
                        while len(h2h_surface_vector) < 4:
                            h2h_surface_vector.append(0)
                    features_vector = [tournament_features_vector, 
                                    player1_features_vector, 
                                    player2_features_vector, 
                                    h2h_overall_vector, 
                                    h2h_surface_vector, 
                                    shape_overall_player1, 
                                    shape_overall_player2
                                    ]
                    features_vectors.append(features_vector)
                    if match_data["result"] != {} :
                        if match_data["result"]["status"] == "Finished" :
                            sets_player1 = int(match_data["result"]["player1"])
                            sets_player2 = int(match_data["result"]["player2"])
                            if sets_player1 > sets_player2 :
                                label_vector.append(1)
                            else :
                                label_vector.append(-1)
                        else :
                            label_vector.append(c.PADDING)
                    else :
                        label_vector.append(c.PADDING)
                    lst_match_id.append(match_id)
                except Exception as e:
                    num_errors += 1
                    tqdm.write(f"Error {e} for match {match_id} in tournament {tournament}")
        return features_vectors, label_vector, lst_match_id, num_errors
    