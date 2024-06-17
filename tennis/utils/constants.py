import os 

curr_dir = os.getcwd()
# if curr_dir.endswith("betbot_tennis") it is ok
if curr_dir.endswith("betbot_tennis"):
    REPO_PATH = curr_dir
else:
    REPO_PATH = ''
    raise Exception("Please run this script from the root of the repository")

# 1 for gran slams 2 for master 1000 3 for ATP 500 4 for ATP 250
TOURNAMENTS_TYPE = {'french-open': 1, 
                    'us-open': 1,
                    'wimbledon': 1,
                    'australian-open': 1,
                    'indian-wells': 2,
                    'miami': 2,
                    'doha': 2,
                    'dubai': 2,
                    'acapulco': 3,
                    'rio-de-janeiro': 3,
                    'santiago': 3,
                    'brisbane': 3,
                    'hong-kong': 3,
                    'auckland': 3,
                    'adelaide': 3,
                    'montpellier': 3,
                    'marseille': 3,
                    'cordoba': 3,
                    'dallas': 3,
                    'monte-carlo': 2,
                    'madrid': 2,
                    'rome': 2,
                    'barcelona': 3,
                    'lyon': 3,
                    'munich': 3,
                    'geneva': 3,
                    'stuttgart': 3,
                    'estoril': 4,
                    'houston': 4,
                    'marrakech': 4,
                    'bucharest': 4,
                    'french-open-2023': 1,
                    'us-open-2023': 1,
                    'wimbledon-2023': 1,
                    'australian-open-2023': 1,
                    'indian-wells-2023': 2,
                    'miami-2023': 2,
                    'doha-2023': 2,
                    'dubai-2023': 2,
                    'acapulco-2023': 3,
                    'rio-de-janeiro-2023': 3,
                    'santiago-2023': 3,
                    'brisbane-2023': 3,
                    'hong-kong-2023': 3,
                    'auckland-2023': 3,
                    'adelaide-2023': 3,
                    'montpellier-2023': 3,
                    'marseille-2023': 3,
                    'cordoba-2023': 3,
                    'dallas-2023': 3,
                    'rome-2023': 2,
                    'madrid-2023': 2,
                    'monte-carlo-2023': 2,
                    'lyon-2023': 3,
                    'barcelona-2023': 3,
                    'munich-2023': 3,
                    'geneva-2023': 3,
                    'estoril-2023': 4,
                    'houston-2023': 4,
                    'marrakech-2023': 4,
                    'bucharest-2023': 4,
                    'stuttgart-2023': 3,
                    'hertogenbosch-2023': 3,
                    'halle-2023': 3,
                    'london-2023': 3,
                    'mallorca-2023': 3,
                    'eastbourne-2023': 3,
                    'wimbledon-2023': 1,
                    'hamburg-2023': 3,
                    'bastad-2023': 3,
                    'gstaad-2023': 3,
                    'newport-2023': 3,
                    'umag-2023': 3,
                    'atlanta-2023': 3,
                    'washington-2023': 3,
                    'montreal-2023': 2,
                    'cincinnati-2023': 2,
                    'winston-salem-2023': 3,
                    'us-open-2023': 1,
                    'chengdu-2023': 3,
                    'zhuhai-2023': 3,
                    'tokyo-2023': 3,
                    'beijing-2023': 3,
                    'shanghai-2023': 2,
                    'stockholm-2023': 3,
                    'antwerp-2023': 3,
                    'almaty-2023': 3,
                    'vienna-2023': 2,
                    'basel-2023': 2,
                    'paris-2023': 2,
                    'metz-2023': 3,
                    'finals-turin-2023': 2,
                    'stuttgart': 3,
                    'hertogenbosch': 3,
}
SURFACE_TYPE = {'clay': 1, 'hard': 2, 'grass': 3}
TENNIS_DATA_PATH = f"{REPO_PATH}/tennis/data/files/matches"
PLAYERS_IDS_AND_NAMES_PATH = f"{REPO_PATH}/tennis/data/files/players_ids.json"
CORRESPONDANCE_FR_IDS_ATP_IDS_PATH = f"{REPO_PATH}/tennis/data/files/fr_to_atp_ids.json"
PLAYERS_DATA_PATH = '/users/eleves-b/2021/mathias.grau/betbot/FlashscoreScraping/src/data/tennis/players_data.json'
TYPE_PLAY = {'Right-Handed, One-Handed Backhand': 0,
 'Right-Handed, Two-Handed Backhand': 1,
 'Left-Handed, Two-Handed Backhand': 2,
 'Left-Handed, One-Handed Backhand': 3,
 'Right-Handed, Unknown Backhand': 4,
 'Left-Handed, Unknown Backhand': 5,}

TOURNAMENTS_SURFACE = {'french-open': 'clay',
                    'us-open': 'hard',
                    'wimbledon': 'grass',
                    'australian-open': 'hard',
                    'indian-wells': 'hard',
                    'miami': 'hard',
                    'doha': 'hard',
                    'dubai': 'hard',
                    'brisbane': 'hard',
                    'hong-kong': 'hard',
                    'auckland': 'hard',
                    'adelaide': 'hard',
                    'montpellier': 'hard',
                    'marseille': 'hard',
                    'cordoba': 'hard',
                    'dallas': 'hard',
                    'acapulco': 'clay',
                    'rio-de-janeiro': 'clay',
                    'santiago': 'clay',
                    'monte-carlo': 'clay',
                    'madrid': 'clay',
                    'rome': 'clay',
                    'barcelona': 'clay',
                    'lyon': 'clay',
                    'munich': 'clay',
                    'geneva': 'clay',
                    'stuttgart': 'clay',
                    'estoril': 'clay',
                    'houston': 'clay',
                    'marrakech': 'clay',
                    'bucharest': 'clay',
                    'french-open-2023': 'clay',
                    'us-open-2023': 'hard',
                    'wimbledon-2023': 'grass',
                    'australian-open-2023': 'hard',
                    'indian-wells-2023': 'hard',
                    'miami-2023': 'hard',
                    'doha-2023': 'hard',
                    'dubai-2023': 'hard',
                    'brisbane-2023': 'hard',
                    'hong-kong-2023': 'hard',
                    'auckland-2023': 'hard',
                    'adelaide-2023': 'hard',
                    'montpellier-2023': 'hard',
                    'marseille-2023': 'hard',
                    'cordoba-2023': 'hard',
                    'dallas-2023': 'hard',
                    'acapulco-2023': 'clay',
                    'rio-de-janeiro-2023': 'clay',
                    'santiago-2023': 'clay',
                    'rome-2023': 'clay',
                    'madrid-2023': 'clay',
                    'monte-carlo-2023': 'clay',
                    'lyon-2023': 'clay',
                    'barcelona-2023': 'clay',
                    'munich-2023': 'clay',
                    'geneva-2023': 'clay',
                    'estoril-2023': 'clay',
                    'houston-2023': 'clay',
                    'marrakech-2023': 'clay',
                    'bucharest-2023': 'clay',
                    'stuttgart-2023': 'grass',
                    'hertogenbosch-2023': 'grass',
                    'halle-2023': 'grass',
                    'london-2023': 'grass',
                    'mallorca-2023': 'grass',
                    'eastbourne-2023': 'grass',
                    'wimbledon-2023': 'grass',
                    'hamburg-2023': 'clay',
                    'bastad-2023': 'clay',
                    'gstaad-2023': 'clay',
                    'newport-2023': 'grass',
                    'umag-2023': 'clay',
                    'atlanta-2023': 'hard',
                    'washington-2023': 'hard',
                    'montreal-2023': 'hard',
                    'cincinnati-2023': 'hard',
                    'winston-salem-2023': 'hard',
                    'us-open-2023': 'hard',
                    'chengdu-2023': 'hard',
                    'zhuhai-2023': 'hard',
                    'tokyo-2023': 'hard',
                    'beijing-2023': 'hard',
                    'shanghai-2023': 'hard',
                    'stockholm-2023': 'hard',
                    'antwerp-2023': 'hard',
                    'almaty-2023': 'hard',
                    'vienna-2023': 'hard',
                    'basel-2023': 'hard',
                    'paris-2023': 'hard',
                    'metz-2023': 'hard',
                    'finals-turin-2023': 'hard',
                    'stuttgart': 'grass',
                    'hertogenbosch': 'grass',
}

PADDING = 0
MAX_MATCHES_FORM = 15




