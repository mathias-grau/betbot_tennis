import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import os
import json

async def get_player_ids():
    url = 'https://www.atptour.com/en/rankings/singles?RankRange=0-5000&Region=all&DateWeek=Current%20Week'
    browser = await launch(headless=True)
    page = await browser.newPage()
    
    # Set user agent
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    await page.goto(url, {'waitUntil': 'networkidle2'})

    # Wait for the necessary content to load
    await page.waitForSelector(".mega-table.desktop-table.non-live")

    content = await page.content()
    await browser.close()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    
    players_ids = {}
    players_names = {}
    
    desktop_player = soup.select_one(".mega-table.desktop-table.non-live")
    if not desktop_player:
        print("desktop_player not found")
        return None

    desktop_player_rows = desktop_player.find_all("tr")

    # Process each player row
    for player_row in desktop_player_rows:
        player_link = player_row.select_one("td.player .player-stats a")
        if player_link:
            href = player_link['href']
            parts = href.split("/")
            if len(parts) >= 5:
                playerId = parts[4]
                fullname = parts[3]
                player_name_tag = player_row.select_one("td.player .player-stats .name")
                if player_name_tag:
                    playerName = player_name_tag.get_text(strip=True)
                    players_ids[playerId] = {
                        'fullname': fullname,
                        'playerName': playerName,
                        'href': href
                    }
                    players_names[playerName] = {
                        'playerId': playerId,
                        'fullname': fullname
                    }
    all_players = {'players_ids': players_ids, 'players_names': players_names}
    return all_players

# Run the coroutine and await it
players_ids = asyncio.run(get_player_ids())

# save it to a json file in /users/eleves-b/2021/mathias.grau/betbot/FlashscoreScraping/src/data/tennis/players_ids.json
with open('/users/eleves-b/2021/mathias.grau/betbot_tennis/tennis/data/files/players_ids.json', 'w') as f:
    json.dump(players_ids, f, indent=4)