import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm
import re
import time

BASE_URL = "https://www.flashscore.com"

# Adjusted scrolling mechanism to scroll until no more matches are loaded
async def get_match_id_list(tournament, league):
    url = f"{BASE_URL}/tennis/{league}/{tournament}/results/"
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        await page.goto(url, {"waitUntil": "networkidle2"})
        
        async def load_more_matches():
            while True:
                previous_height = await page.evaluate('document.body.scrollHeight')
                try:
                    await page.waitForSelector(".event__more.event__more--static", timeout=2000)
                    await page.focus(".event__more.event__more--static")
                    await page.keyboard.type('\n')
                    await asyncio.sleep(1)  # Adjust as necessary to allow new content to load

                    new_height = await page.evaluate('document.body.scrollHeight')
                    if new_height == previous_height:
                        print("breaking")
                        break
                except asyncio.TimeoutError:
                    break
        await load_more_matches()
        
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        all_matches = soup.select(".event__match.event__match--static.event__match--twoLine")
        for match in all_matches:
            # map(element => element?.id?.replace("g_2_", ""))
            match_id = match.get("id").replace("g_2_", "")
        print(f"Found {len(all_matches)} matches.")
        return all_matches
    
    finally:
        await browser.close()

async def get_match_data(tournament, league, match_id):
    base_url = 'https://www.flashscore.com/match/'
    url_statistics = f'{base_url}{match_id}/#/match-summary/match-statistics/0'
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        await page.goto(url_statistics, {"waitUntil": "networkidle2"})
        await page.waitForSelector(".duelParticipant__home")
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        
        date = soup.select_one(".duelParticipant__startTime").text
        players = {
            "player1": {
                "name": soup.select_one(".duelParticipant__home .participant__participantName.participant__overflow").text,
                "href": soup.select_one(".duelParticipant__home .participant__participantName.participant__overflow a").get("href"),
                "fullname": soup.select_one(".duelParticipant__home .participant__participantName.participant__overflow a").get("href").split("/")[2],
                "id": soup.select_one(".duelParticipant__home .participant__participantName.participant__overflow a").get("href").split("/")[3],
                "image": soup.select_one(".duelParticipant__home .participant__image").get("src")
            },
            "player2": {
                "name": soup.select_one(".duelParticipant__away .participant__participantName.participant__overflow").text,
                "href": soup.select_one(".duelParticipant__away .participant__participantName.participant__overflow a").get("href"),
                "fullname": soup.select_one(".duelParticipant__away .participant__participantName.participant__overflow a").get("href").split("/")[2],
                "id": soup.select_one(".duelParticipant__away .participant__participantName.participant__overflow a").get("href").split("/")[3],
                "image": soup.select_one(".duelParticipant__away .participant__image").get("src")
            }
        }

        if soup.select_one(".fixedHeaderDuel__detailStatus").text == 'Walkover' : 
            return None

        result = {
            "player1": soup.select(".detailScore__wrapper span:not(.detailScore__divider)")[0].text,
            "player2": soup.select(".detailScore__wrapper span:not(.detailScore__divider)")[1].text,
            "status": soup.select_one(".fixedHeaderDuel__detailStatus").text,
        }

        statistics = []
        for element in soup.select("div[data-testid='wcl-statistics']"):
            statistics.append({
                "categoryName": element.select_one("div[data-testid='wcl-statistics-category']").text,
                "player1Value": element.select("div[data-testid='wcl-statistics-value'] > strong")[0].text,
                "player2Value": element.select("div[data-testid='wcl-statistics-value'] > strong")[1].text,
            })
        
        data = {
            "date": date,
            "players": players,
            "result": result,
            "statistics": statistics,
            "tournament": tournament,
            "league": league
        }
        
        # Fetch score data
        url_score = f'{base_url}{match_id}/#/match-summary'
        await page.goto(url_score, {"waitUntil": "networkidle2"})
        await asyncio.sleep(1)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        
        score_element = soup.select_one(".smh__part")
        if score_element:
            player1_sets = [int(set_.text) for set_ in soup.select(".smh__part.smh__home.smh__part:not(.smh__home.smh__part--current)") if set_.text.isdigit()]
            player2_sets = [int(set_.text) for set_ in soup.select(".smh__part.smh__away.smh__part:not(.smh__away.smh__part--current)") if set_.text.isdigit()]
            data["score"] = {"player1Sets": player1_sets, "player2Sets": player2_sets}
        else:
            data["score"] = None
        
        # Fetch head-to-head data
        for surface, suffix in [("h2h_overall", "overall"), ("h2h_clay", "home"), ("h2h_grass", "away"), ("h2h_hard", "3")]:
            url_h2h = f'{base_url}{match_id}/#/h2h/{suffix}'
            await page.goto(url_h2h, {"waitUntil": "networkidle2"})
            await asyncio.sleep(1)
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            
            h2h_data = []
            for section in soup.select(".h2h__section.section"):
                title = section.select_one(".section__title").text
                rows = section.select(".h2h__row")
                data_rows = []
                for row in rows:
                    date = row.select_one(".h2h__date").text
                    event = row.select_one(".h2h__event").text
                    player1 = row.select_one(".h2h__participant.h2h__homeParticipant").text
                    player2 = row.select_one(".h2h__participant.h2h__awayParticipant").text
                    result = row.select_one(".h2h__result").text
                    data_rows.append({
                        "date": date,
                        "event": event,
                        "player1": player1,
                        "player2": player2,
                        "resultPlayer1": result[0] if (result and result!='-') else None,
                        "resultPlayer2": result[1] if (result and result!='-') else None,
                    })
                h2h_data.append({"title": title, "data": data_rows})
            data[surface] = h2h_data
        
        # Fetch odds data
        url_odds = f'{base_url}{match_id}/#/odds-comparison/1x2-odds/full-time'
        await page.goto(url_odds, {"waitUntil": "networkidle2"})
        await asyncio.sleep(1)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        
        odds = []
        if soup.select(".ui-table__row"):
            for row in soup.select(".ui-table__row"):
                bookmaker_element = row.select_one(".oddsCell__bookmaker a")
                bookmaker = bookmaker_element.get("title") if bookmaker_element else "Unknown Bookmaker"
                link = bookmaker_element.get("href") if bookmaker_element else "Unknown Link"
                odds_values = [odd.text.strip() for odd in row.select(".oddsCell__odd")[:3]]
                odds.append({"bookmaker": bookmaker, "link": link, "odds": odds_values}) 
        data["odds"] = odds
        
        return data

    finally:
        await browser.close()

# Example usage
async def main():
    matches = await get_match_id_list("french-open", "atp-singles")
    match_dict = {}
    for match in tqdm(matches):
        match_id = match.get("id").replace("g_2_", "")
        match_data = await get_match_data("french-open", "atp-singles", match_id)
        if match_data:
            match_dict[match_id] = match_data
    with open("matches.json", "w") as f:
        json.dump(match_dict, f, indent=4)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
