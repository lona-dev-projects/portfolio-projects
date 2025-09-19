import asyncio
import json
import logging
import csv
import random
import re
from datetime import datetime
from itertools import combinations
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

BANKROLL = 100.0
MAX_MATCHES = 15 
PROXY_LIST = [
    'http://user:pass@proxy1:port',
    'http://user:pass@proxy2:port',
]
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15...',
]
SITES = {
    'Betway': 'https://betway.co.za/sports/tennis',
    'SportingBet': 'https://sports.sportingbet.co.za/en/sports/tennis-7'
}

SELECTORS = {
    'Betway': {
        'matches_xpath': "//div[contains(translate(., 'VS', 'vs'), ' vs ') or contains(., ' v ')]",
        'players_xpath': ".//div[contains(@class, 'participant') or contains(@class, 'name')]",
        'odds_xpath': ".//div[contains(@class, 'price') or contains(@class, 'odds')]"
    },
    'SportingBet': {
        'matches_xpath': "//div[contains(., ' vs ') or contains(., ' v ')]",
        'players_xpath': ".//div[contains(@class, 'participant')]",
        'odds_xpath': ".//span[contains(@class, 'odds')]"
    }
    
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AntiDetection:
    @staticmethod
    async def human_like_interaction(page):
        await page.mouse.move(
            random.randint(0, 1000),
            random.randint(0, 600)
        )
        await page.wait_for_timeout(random.randint(200, 1500))
        await page.mouse.wheel(0, random.randint(0, 500))

    @staticmethod
    def random_viewport():
        return {
            'width': random.choice([1366, 1920, 1440, 1600]),
            'height': random.choice([768, 1080, 900, 1024])
        }

class AdvancedScraper:
    def __init__(self):
        self.proxy_rotation = iter(PROXY_LIST)
        self.ua_rotation = iter(USER_AGENTS)
    
    async def create_browser(self, playwright):
        proxy_str = next(self.proxy_rotation, None)
        proxy = self.parse_proxy(proxy_str) if proxy_str else None

        return await playwright.chromium.launch(
            headless=True,
            proxy=proxy or None,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                f'--user-agent={next(self.ua_rotation, USER_AGENTS[0])}'
            ]
        )
    
    def parse_proxy(self, proxy_str):
        try:
            match = re.match(r'^(.*?)://(?:([^:]+):([^@]*)@)?(.+)$', proxy_str)
            if not match:
                return None
            
            protocol = match.group(1)
            username = match.group(2)
            password = match.group(3)
            host = match.group(4)

            return {
                'server': f'{protocol}://{host}',
                'username': username,
                'password': password
            }
        except Exception as e:
            logger.error(f"Error parsing proxy {proxy_str}: {e}")
            return None
        
    async def scrape_site(self, page, site_name):
        try:
            config = SELECTORS[site_name]
            await page.set_viewport_size(AntiDetection.random_viewport())
            await page.goto(SITES[site_name], timeout=30000)
            await AntiDetection.human_like_interaction(page)
            
            matches = []
            match_elements = await page.query_selector_all(config['matches_xpath'])
            
            for element in match_elements[:MAX_MATCHES]:  # Limit matches per site
                try:
                    player_elements = await element.query_selector_all(config['players_xpath'])
                    players = ' vs '.join([
                        await self.normalize_name(await el.inner_text())
                        for el in player_elements[:2]
                    ])
                    
                    odds_elements = await element.query_selector_all(config['odds_xpath'])
                    odds = []
                    for el in odds_elements[:2]:
                        try:
                            text = await el.inner_text()
                            odds.append(float(re.search(r'\d+\.?\d*', text).group()))
                        except:
                            continue
                    
                    if len(odds) == 2:
                        matches.append({
                            'players': players,
                            'odds': odds,
                            'site': site_name
                        })
                except Exception as e:
                    logger.error(f"Error processing {site_name} match: {e}")
            return matches
        except PlaywrightTimeoutError as te:
            logger.error(f"Timeout scraping {site_name}: {te}")
            return []
        except Exception as e:
            logger.error(f"Critical error scraping {site_name}: {e}")
            return []

    @staticmethod
    def normalize_name(name: str) -> str:
        name = re.sub(r'[^a-zA-Z\s]', '', name).strip().lower()
        name = re.sub(r'\b(vs?|versus)\b', 'vs', name)
        parts = [p for p in re.split(r'\s+', name) if p not in ['', 'vs']]
        return ' '.join(sorted(parts))
class ArbitrageCalculator:
    @staticmethod
    def generate_arbitrage_report(all_matches):
        report = []
        match_dict = {}
        for site, matches in all_matches.items():
            for match in matches:
                key = match['players']
                if key not in match_dict:
                    match_dict[key] = {}
                match_dict[key][site] = match['odds']

        for match_name, odds_data in match_dict.items():
            try:
                all_player1_odds = []
                all_player2_odds = []
                
                for site, odds in odds_data.items():
                    all_player1_odds.append((site, odds[0]))
                    all_player2_odds.append((site, odds[1]))
                
                best_p1 = max(all_player1_odds, key=lambda x: x[1]) if all_player1_odds else (None, 0)
                best_p2 = max(all_player2_odds, key=lambda x: x[1]) if all_player2_odds else (None, 0)

                arb_sum = (1/best_p1[1] + 1/best_p2[1]) if best_p1[1] and best_p2[1] else 0
                opportunity_exists = arb_sum < 1 if arb_sum else False
                
                if opportunity_exists:
                    stake1 = (BANKROLL / best_p1[1]) / arb_sum
                    stake2 = (BANKROLL / best_p2[1]) / arb_sum
                    profit = BANKROLL / arb_sum - BANKROLL
                    roi = (profit / BANKROLL) * 100
                else:
                    stake1 = stake2 = profit = roi = 0

                report.append({
                    'match': match_name,
                    'p1_odds': best_p1[1],
                    'p1_sites': best_p1[0],
                    'p2_odds': best_p2[1],
                    'p2_sites': best_p2[0],
                    'arb_sum': arb_sum,
                    'opportunity': opportunity_exists,
                    'stake1': stake1,
                    'stake2': stake2,
                    'profit': profit,
                    'roi': roi
                })
                
            except Exception as e:
                logger.error(f"Error processing {match_name}: {e}")
        
        return report
async def main():
    scraper = AdvancedScraper()
    async with async_playwright() as playwright:
        browser = await scraper.create_browser(playwright)
        try:
            context = await browser.new_context(
                user_agent=next(scraper.ua_rotation, USER_AGENTS[0]),
                viewport=AntiDetection.random_viewport()
            )
            
            tasks = []
            for site in SITES:
                page = await context.new_page()
                tasks.append(scraper.scrape_site(page, site))
            
            results = await asyncio.gather(*tasks)
            scraped_data = {site: data for site, data in zip(SITES.keys(), results)}
            all_matches = []
            for site in SITES.keys():
                all_matches.extend(scraped_data.get(site, []))
            selected_matches = all_matches[:MAX_MATCHES]
            limited_data = {site: [] for site in SITES}
            for match in selected_matches:
                limited_data[match['site']].append(match)
            
            calculator = ArbitrageCalculator()
            report = calculator.generate_arbitrage_report(limited_data)
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'arbitrage_report_{timestamp}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Match', 'Player1 Odds', 'Player1 Sites', 'Player2 Odds', 'Player2 Sites',
                    'Arb Sum', 'Opportunity', 'Stake1', 'Stake2', 'Profit', 'ROI%'
                ])
                for entry in report:
                    writer.writerow([
                        entry['match'],
                        round(entry['p1_odds'], 2),
                        entry['p1_sites'],
                        round(entry['p2_odds'], 2),
                        entry['p2_sites'],
                        round(entry['arb_sum'], 4),
                        'Yes' if entry['opportunity'] else 'No',
                        round(entry['stake1'], 2) if entry['opportunity'] else 'N/A',
                        round(entry['stake2'], 2) if entry['opportunity'] else 'N/A',
                        round(entry['profit'], 2) if entry['opportunity'] else 'N/A',
                        round(entry['roi'], 2) if entry['opportunity'] else 'N/A'
                    ])
            
            logger.info(f"Processed {len(report)} matches. Found {sum(1 for r in report if r['opportunity'])} opportunities")
        
        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(main())