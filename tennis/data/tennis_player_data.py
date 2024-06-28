import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
import re
import time
import utils.constants as c

YEAR = 2024
YEARS = [2023,2024]


async def get_player_data(playerId, fullname):
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/overview"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    await page.waitForSelector(".personal_details .pd_content")
    content = await page.content()
    await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    playerData = {
        "playerId": playerId,
        "fullname": fullname
    }
    pdContent = soup.find(class_="personal_details").find(class_="pd_content")
    if not pdContent:
        tqdm.write("pdContent not found")
        return None
    playerData["data"] = {
        "playerName": soup.select_one(".player_name").get_text(strip=True),
        "age": pdContent.select_one(".pd_left li:nth-child(1) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_left li:nth-child(1) span:nth-child(2)") else None,
        "weight": pdContent.select_one(".pd_left li:nth-child(2) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_left li:nth-child(2) span:nth-child(2)") else None,
        "height": pdContent.select_one(".pd_left li:nth-child(3) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_left li:nth-child(3) span:nth-child(2)") else None,
        "turnedPro": pdContent.select_one(".pd_left li:nth-child(4) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_left li:nth-child(4) span:nth-child(2)") else None,
        "socialMediaLinks": [
            {
                "platform": a.find("span")["class"][0].replace("icon-", "") if a.find("span") else None,
                "url": a["href"]
            } for a in pdContent.select(".pd_left .social ul li a")
        ],
        "country": pdContent.select_one(".pd_right li:nth-child(1) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_right li:nth-child(1) span:nth-child(2)") else None,
        "birthplace": pdContent.select_one(".pd_right li:nth-child(2) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_right li:nth-child(2) span:nth-child(2)") else None,
        "typePlays": pdContent.select_one(".pd_right li:nth-child(3) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_right li:nth-child(3) span:nth-child(2)") else None,
        "coach": pdContent.select_one(".pd_right li:nth-child(4) span:nth-child(2)").get_text(strip=True) if pdContent.select_one(".pd_right li:nth-child(4) span:nth-child(2)") else None,
    }
    playerData["href"] = url

    playerData["statistics"] = {}
    # statistics on all surfaces 
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-stats?year=all&surface=all"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    await page.waitForSelector(".statistics_content")
    content = await page.content()
    await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    stats_content = soup.find(class_="statistics_content")
    if stats_content:
        serve_stats = {}
        return_stats = {}
        serve_section = stats_content.find_all("div")[0]
        serve_stats_items = serve_section.find_all(class_="stats_items")
        for item in serve_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            serve_stats[record] = percentage
        return_section = stats_content.find_all("div")[1]
        return_stats_items = return_section.find_all(class_="stats_items")
        for item in return_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            return_stats[record] = percentage
        playerData["statistics"]["all"] = {
            "serve": serve_stats,
            "return": return_stats
        }
    else:
        tqdm.write("statistics_content for all surfaces not found")
    # statistics on hard surface
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-stats?year=all&surface=Hard"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    await page.waitForSelector(".statistics_content")
    content = await page.content()
    await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    stats_content = soup.find(class_="statistics_content")
    if stats_content:
        serve_stats = {}
        return_stats = {}
        serve_section = stats_content.find_all("div")[0]
        serve_stats_items = serve_section.find_all(class_="stats_items")
        for item in serve_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            serve_stats[record] = percentage
        return_section = stats_content.find_all("div")[1]
        return_stats_items = return_section.find_all(class_="stats_items")
        for item in return_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            return_stats[record] = percentage
        playerData["statistics"]["hard"] = {
            "serve": serve_stats,
            "return": return_stats
        }
    else:
        tqdm.write("statistics_content for hard surface not found")
    # statistics on clay surface
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-stats?year=all&surface=Clay"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    await page.waitForSelector(".statistics_content")
    content = await page.content()
    await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    stats_content = soup.find(class_="statistics_content")
    if stats_content:
        serve_stats = {}
        return_stats = {}
        serve_section = stats_content.find_all("div")[0]
        serve_stats_items = serve_section.find_all(class_="stats_items")
        for item in serve_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            serve_stats[record] = percentage
        return_section = stats_content.find_all("div")[1]
        return_stats_items = return_section.find_all(class_="stats_items")
        for item in return_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            return_stats[record] = percentage
        playerData["statistics"]["clay"] = {
            "serve": serve_stats,
            "return": return_stats
        }
    else:
        tqdm.write("statistics_content for clay surface not found")
    # statistics on grass surface
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-stats?year=all&surface=Grass"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    await page.waitForSelector(".statistics_content")
    content = await page.content()
    await browser.close()
    soup = BeautifulSoup(content, "html.parser")
    stats_content = soup.find(class_="statistics_content")
    if stats_content:
        serve_stats = {}
        return_stats = {}
        serve_section = stats_content.find_all("div")[0]
        serve_stats_items = serve_section.find_all(class_="stats_items")
        for item in serve_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            serve_stats[record] = percentage
        return_section = stats_content.find_all("div")[1]
        return_stats_items = return_section.find_all(class_="stats_items")
        for item in return_stats_items:
            record = item.find(class_="stats_record").get_text(strip=True)
            percentage = item.find(class_="stats_percentage").get_text(strip=True)
            return_stats[record] = percentage
        playerData["statistics"]["grass"] = {
            "serve": serve_stats,
            "return": return_stats
        }
    else:
        tqdm.write("statistics_content for grass surface not found")
    # activity win loss
    url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/atp-win-loss?tourType=Tour"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    await page.goto(url, {"waitUntil": "networkidle2"})
    try : 
        await page.waitForSelector(".scrollable-table")
        content = await page.content()
        await browser.close()
        soup = BeautifulSoup(content, "html.parser")
        match_record_table = soup.find(class_="scrollable-table")
        titles = ["ytd_wl", "ytd_index", "career_wl", "career_index", "career_titles"]
        if match_record_table:
            match_records = {}
            table_rows = match_record_table.find("tbody").find_all("tr")
            for row in table_rows:
                category = row.find("th")
                # if category is class main-header, skip
                if category == "main-header":
                    continue
                type_category = category.get_text(strip=True)
                cells = row.find_all("td")
                match_records[type_category] = {}

                for i, cell in enumerate(cells) : 
                    match_records[type_category][titles[i]] = cell.get_text(strip=True)

            playerData["match_records"] = match_records
        else:
            tqdm.write("scrollable-table not found")
    except :
        tqdm.write("scrollable-table not found")
        playerData["match_records"] = {}
    
    try :
        all_results = {}
        for year in YEARS:
            all_results[year] = {}
            results_year = {}
            url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-activity?matchType=Singles&={year}4&tournament=all"
            browser = await launch(headless=True)
            page = await browser.newPage()
            await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            await page.goto(url, {"waitUntil": "networkidle2"})
            await page.waitForSelector(".tournament.tournament--expanded")
            content = await page.content()
            await browser.close()
            soup = BeautifulSoup(content, "html.parser")
            # select all tournament tournament--expanded
            all_tournaments = soup.select(".tournament.tournament--expanded")
            for tournament in all_tournaments:
                results_tournament = {}
                tournament_name = tournament.select_one(".status-country").text.strip()
                results_year[tournament_name] = {}
                tournament_location = tournament.select_one(".date-location").text.split("|")[0].strip()
                results_tournament["location"] = tournament_location
                tournament_date = tournament.select_one(".date-location").text.split("|")[1].strip()
                results_tournament["date"] = tournament_date
                tournament_surface = tournament.select_one(".date-location").text.split("|")[2].strip()
                results_tournament["surface"] = tournament_surface
                list_matches = tournament.select(".ranking-item")
                matches = []
                for i, match in enumerate(list_matches):
                    # moment in <dt> format

                    result_match = {}
                    opponent_name = match.select_one(".name").text.split("(")[0].strip()
                    result_match["opponent_name"] = opponent_name
                    if opponent_name == "Bye":
                        # skip the match 
                        continue
                    opponent_rank = match.select_one(".op-rank").text.strip()
                    opponent_rank = int(re.sub(r"\D", "", opponent_rank))
                    result_match["opponent_rank"] = opponent_rank
                    match_result = match.select_one(".set-check")
                    # if span class icon-cross is present, the player lost the match else if span class icon-check is present, the player won the match
                    if match_result.select_one(".icon-cross"):
                        match_result = -1
                    elif match_result.select_one(".icon-checkmark"):
                        match_result = 1
                    else:
                        match_result = None
                    result_match["result"] = match_result
                    matches.append(result_match)
                results_tournament['matches'] = matches
                results_year[tournament_name] = results_tournament
            all_results[year] = results_year
        playerData["results"] = all_results
    except :
        time.sleep(5)
        tqdm.write("player shape not found, retrying")
        all_results = {}
        for year in YEARS:
            all_results[year] = {}
            results_year = {}
            url = f"https://www.atptour.com/en/players/{fullname}/{playerId}/player-activity?matchType=Singles&={year}4&tournament=all"
            browser = await launch(headless=True)
            page = await browser.newPage()
            await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            await page.goto(url, {"waitUntil": "networkidle2"})
            await page.waitForSelector(".tournament.tournament--expanded")
            content = await page.content()
            await browser.close()
            soup = BeautifulSoup(content, "html.parser")
            # select all tournament tournament--expanded
            all_tournaments = soup.select(".tournament.tournament--expanded")
            for tournament in all_tournaments:
                results_tournament = {}
                tournament_name = tournament.select_one(".status-country").text.strip()
                results_year[tournament_name] = {}
                tournament_location = tournament.select_one(".date-location").text.split("|")[0].strip()
                results_tournament["location"] = tournament_location
                tournament_date = tournament.select_one(".date-location").text.split("|")[1].strip()
                results_tournament["date"] = tournament_date
                tournament_surface = tournament.select_one(".date-location").text.split("|")[2].strip()
                results_tournament["surface"] = tournament_surface
                list_matches = tournament.select(".ranking-item")
                matches = []
                for i, match in enumerate(list_matches):
                    # moment in <dt> format

                    result_match = {}
                    opponent_name = match.select_one(".name").text.split("(")[0].strip()
                    result_match["opponent_name"] = opponent_name
                    if opponent_name == "Bye":
                        # skip the match 
                        continue
                    opponent_rank = match.select_one(".op-rank").text.strip()
                    opponent_rank = int(re.sub(r"\D", "", opponent_rank))
                    result_match["opponent_rank"] = opponent_rank
                    match_result = match.select_one(".set-check")
                    # if span class icon-cross is present, the player lost the match else if span class icon-check is present, the player won the match
                    if match_result.select_one(".icon-cross"):
                        match_result = -1
                    elif match_result.select_one(".icon-checkmark"):
                        match_result = 1
                    else:
                        match_result = None
                    result_match["result"] = match_result
                    matches.append(result_match)
                results_tournament['matches'] = matches
                results_year[tournament_name] = results_tournament
            all_results[year] = results_year
        playerData["results"] = all_results

    return playerData
    

# Path to player IDs file
player_ids_file = f'{c.REPO_PATH}/tennis/data/files/players_ids.json'

# Path to player data file
player_data_file = f'{c.REPO_PATH}/tennis/data/files/players_data.json'

# Load player IDs
with open(player_ids_file, "r") as f:
    players_ids = json.load(f)

# Load existing player data if the file exists, otherwise initialize an empty dictionary
if os.path.exists(player_data_file):
    with open(player_data_file, "r") as f:
        all_players_data = json.load(f)
else:
    all_players_data = {}

# Get player data
players_ids = players_ids["players_ids"]

# Iterate over player IDs and fetch data if not already present
for i, playerId in tqdm(enumerate(players_ids), total=len(players_ids)):
    player = players_ids[playerId]
    if playerId not in all_players_data:
        fullname = player["fullname"]
        tqdm.write(f"{i+1} : {fullname}")
        player_data = asyncio.run(get_player_data(playerId, fullname))
        if player_data:
            all_players_data[playerId] = player_data

            # Save the updated player data after each new addition
            with open(player_data_file, "w") as f:
                json.dump(all_players_data, f, indent=4)
    else:
        tqdm.write(f"{i+1} : {player['fullname']} already exists")

tqdm.write("All player data has been updated and saved.")