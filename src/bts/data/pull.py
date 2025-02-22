import re
import urllib
import xml.etree.ElementTree
import datetime
import os
import pdb
import argparse
import sys
import pandas as pd
from pybaseball import statcast
from pybaseball.retrosheet import events
import argparse
from meteostat import Hourly, Point
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import glob
import shutil
from IPython import embed

def download_statcast(start, end, base):
    for day in pd.date_range(start, end):
        daystr = day.strftime('%Y-%m-%d')
        print(daystr)
        data = statcast(daystr)

        if not os.path.isdir(base + '/' + daystr):
            os.makedirs(base + '/' + daystr)

        data.to_csv('%s/%s/statcast.csv' % (base, daystr), index=False)

def download_retrosheet(start, end, base):
    # Note: if this fails, just download the retrosheet data manually
    loc = base + '/retrosheet/'
    for year in range(start.year, end.year+1):
        events(year, export_dir=loc)


def download_weather(start, end, base):
    locs = {}
    locs['SF'] = (37.7765057,-122.4060536)
    locs['LAA'] = (33.8002688,-117.885124)
    locs['STL'] = (38.6226188,-90.1928209)
    locs['ARI'] = (33.4453389,-112.0689031)

    locs['NYM'] = (40.7570917,-73.8480153)
    locs['PHI'] = (39.9060613,-75.1686892)
    locs['DET'] = (42.3390023,-83.0507137)
    locs['COL'] = (39.7558864,-104.9963721)

    locs['LAD'] = (34.0738554,-118.2421523)
    locs['BOS'] = (42.3466803,-71.0994118)
    locs['CIN'] = (39.0973849,-84.5092654)
    locs['KC'] = (39.051676,-94.4825082)

    locs['MIL'] = (43.0279815,-87.9733444)
    locs['HOU'] = (29.757274,-95.3577083)
    locs['WAS'] = (38.8730144,-77.0096269)
    locs['OAK'] = (37.7509705,-122.2039914)

    locs['BAL'] = (39.2838235,-76.6238722)
    locs['SD'] = (32.7072738,-117.158939)
    locs['PIT'] = (40.4469451,-80.0078994)
    locs['CLE'] = (41.496215,-81.6874229)

    locs['TEX'] = (32.7492374,-97.08451)
    locs['TOR'] = (43.6417837,-79.3913377)
    locs['SEA'] = (47.5914062,-122.3347025)
    locs['MIA'] = (25.7781487,-80.2371093)

    locs['MIN'] = (44.9802399,-93.2801412)
    locs['TB'] = (27.7682293,-82.6555861)
    locs['ATL'] = (33.891069,-84.4706438)
    locs['CWS'] = (41.8299061,-87.6359462)

    locs['CHC'] = (41.9484133,-87.6912316)
    locs['NYY'] = (40.8296466,-73.9283685)

    alts = {'TEX': 187.74763791526973,
             'LAA': 48.7656202377324,
             'BAL': 39.62206644315757,
             'DET': 181.65193538555317,
             'BOS': 6.09570252971655,
             'CLE': 177.3849436147516,
             'KC': 228.5888448643706,
             'OAK': 12.800975312404754,
             'MIN': 247.4855227064919,
             'TOR': 75.28192624199939,
             'SEA': 3.047851264858275,
             'TB': 13.41054556537641,
             'CWS': 181.65193538555317,
             'NYY': 16.458396830234683,
             'SF': 19.20146296860713,
             'STL': 138.6772325510515,
             'ARI': 329.77750685766534,
             'PHI': 2.7430661383724475,
             'COL': 1579.7013105760439,
             'LAD': 81.37762877171593,
             'MIA': 4.571776897287412,
             'CIN': 208.16824138982017,
             'MIL': 180.7375800060957,
             'HOU': 11.581834806461444,
             'SD': 3.9622066443157573,
             'PIT': 226.45534897896982,
             'WAS': 7.619628162145687,
             'NYM': 16.458396830234683,
             'ATL': 320.0243828101189,
             'CHC': 181.65193538555317}

    weather = []
    for loc in locs.keys():
        print('Processing', loc)
        p = Point(*locs[loc], alts[loc])
        tz = p.get_stations()['timezone'].iloc[0]
        stats = Hourly(p, start, end, timezone=tz)
        data = stats.fetch()
        data['ballpark'] = loc
        data['lat'] = locs[loc][0]
        data['lon'] = locs[loc][1]
        data['alt'] = alts[loc]
        data['timezone'] = tz
        weather.append(data)

    data = pd.concat(weather)
    data.to_csv(base + '/weather.csv')

def download_historical_vegas(start, end, base):
    # Pretty buggy, usually fails after 25 steps, have to manually rerun each time it fails
    key = show_date(start - datetime.timedelta(days=1))
    url = 'https://www.bettingpros.com/mlb/odds/player-props/to-record-a-hit/?date=%s' % key
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--headless=new')  # disable to debug
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.maximize_window()
    driver.implicitly_wait(5)

    results = []
    NEXT_XPATH = '//button[@class="button button--styleless date-picker__arrow date-picker__arrow--right"]'
    NAME_XPATH = '//div[@class="odds-player__info odds-player__info--small"]'
    OPEN_XPATH = '//div[@class="odds-cell odds-cell--open odds-cell--event-completed"]'
    CONSENSUS_XPATH = '//button[@class="odds-cell odds-cell--default odds-cell--event-completed"]'
    for day in pd.date_range(start, end):
        driver.find_element(By.XPATH, NEXT_XPATH).click()
        
        names = []
        teams = []
        open_odds = []
        closed_odds = []
        for element in driver.find_elements(By.XPATH, NAME_XPATH):
            split = element.text.split('\n')
            name = split[0]
            team = split[1].split('-')[0][:-1]
            names.append(name)
            teams.append(team)
            
        for element in driver.find_elements(By.XPATH, OPEN_XPATH):
            open_odds.append(element.text)
        
        for element in driver.find_elements(By.XPATH, CONSENSUS_XPATH):
            closed_odds.append(element.text)
       
        try:
            df = pd.DataFrame()
            df['name'] = names
            df['team'] = teams
            df['date'] = day
            df['open_odds'] = open_odds
            df['closed_odds'] = closed_odds
            daystr = show_date(day)
            df.to_csv('%s/%s/vegas.csv' % (base, daystr), index=False)
            print('Downloaded Vegas Data', day)
        except Exception as e:
            print('Skipping Vegas Data', day)
            print(e)
            continue

    driver.close()


def download_vegas(base):

    daystr = show_date(pd.Timestamp.today())
    timestr = pd.Timestamp.today().strftime('%Y-%m-%d_%H:%M')

    url = 'https://www.bettingpros.com/mlb/odds/player-props/to-record-a-hit/?date=%s' % daystr
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--headless=new')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    #driver.maximize_window()
    driver.implicitly_wait(5)

    results = []
    NAME_XPATH = '//div[@class="odds-player__info odds-player__info--small"]'
    OFFER_XPATH = '//div[@class="flex odds-offer__item"]'
    HEADER_XPATH = '//div[@class="flex odds-offers-header__item"]/img'
        
    names = []
    teams = []
    offers = []
    header = []

    for element in driver.find_elements(By.XPATH, HEADER_XPATH):
        column = element.get_attribute('alt').strip('Logo for ')
        header.append(column)

    header.append('Consensus')

    # TODO: might it be faster to parse the entire block at once
    # as we do for the offer cells?
    for element in driver.find_elements(By.XPATH, NAME_XPATH):
        split = element.text.split('\n')
        name = split[0]
        team = split[1].split('-')[0][:-1]
        names.append(name)
        teams.append(team)
        
    for element in driver.find_elements(By.XPATH, OFFER_XPATH):
        offers.append(element.text)

    driver.close()

    offer_table = [x.split('\n') for x in offers]
    assert len(offer_table) % len(header) == 0

    df = pd.DataFrame()
    df['name'] = names
    df['team'] = teams
    df['date'] = daystr
    for index, column in enumerate(header):
        df[column] = sum(offer_table[index::len(header)], [])

    path = os.path.join(base, daystr, 'vegas_%s.csv' % timestr) 
    df.to_csv(path, index=False)

    convenient_loc = os.path.join(os.environ['BTSHOME'], 'vegas.csv')
    shutil.copy(path, convenient_loc)



def download_projections(base):
    """Download projections from external model the current day."""
    options = webdriver.ChromeOptions()
    # Note: this option seems to work on my Desktop, but not on my laptop
    # We must use headless mode to be compatible with crontab
    options.add_argument('--headless=new')
    driver = webdriver.Chrome(options=options)
    username = os.environ['ROTOGRINDERS_USERNAME']
    password = os.environ['ROTOGRINDERS_PASSWORD']
    driver.get('https://rotogrinders.com/sign-in')

    driver.maximize_window()
    driver.implicitly_wait(5)
    username_field = driver.find_element(By.NAME, 'username')
    username_field.send_keys(username)
    password_field = driver.find_element(By.NAME, 'password')
    password_field.send_keys(password)
    sign_in_xpath = "//input[@class='button highlight cta' and @type='submit']"
    driver.find_element(By.XPATH, sign_in_xpath).click()

    links = ['https://rotogrinders.com/grids/standard-projections-the-bat-x-hitters-3372512',
            'https://rotogrinders.com/grids/standard-projections-the-bat-x-3372510']
    xpaths = ["//a[@data-role='linkable' and @data-pointer='L2dyaWRzLzMzNzI1MTIuY3N2']",
            "//a[@data-role='linkable' and @data-pointer='L2dyaWRzLzMzNzI1MTAuY3N2']"]
    names = ["batter_projections", "pitcher_projections"]

    daystr = show_date(pd.Timestamp.today())
    timestr = pd.Timestamp.today().strftime('%Y-%m-%d_%H:%M')
    if not os.path.isdir(base + '/' + daystr):
        os.makedirs(base + '/' + daystr)

    for link, xpath, name in zip(links, xpaths, names):

        driver.get(link)
        element = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        #driver.find_element(By.XPATH, DOWNLOAD_XPATH).click()
        element.click()
        time.sleep(5)

        files = glob.glob(os.path.join(os.environ['HOME'], 'Downloads/*.csv'))
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)

        loc = os.path.join(base, daystr, name + '_' + timestr + '.csv')
        shutil.move(sorted_files[0], loc)
    
        ironcond0r_loc = os.path.join(os.environ['BTSHOME'], name + '.csv')
        shutil.copy(loc, ironcond0r_loc)

    driver.close()


def parse_date(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def show_date(date):
    return date.strftime('%Y-%m-%d')

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    yesterday = show_date(pd.Timestamp.today() - datetime.timedelta(days=1))
    params = {}
    params['start'] = yesterday
    params['end'] = yesterday
    params['source'] = 'statcast'

    return params

if __name__ == '__main__':

    description = 'download data and store in appropriate folder'
    formatter = argparse.ArgumentDefaultsHelpFormatter    
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--source', choices=['statcast', 'retrosheet', 'weather', 'vegas', 'projections'], help='data source to download')
    parser.add_argument('--start', help='start date (yyyy-mm-dd)')
    parser.add_argument('--end', help='end date (yyyy-mm-dd)')

    base = os.environ['BTSDATA']

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    if args.source == 'statcast':
        download_statcast(start, end, base)
    if args.source == 'retrosheet':
        download_retrosheet(start, end, base)
    if args.source == 'weather':
        download_weather(start, end, base)
    if args.source == 'historical-vegas':
        download_historical_vegas(start, end, base)
    if args.source == 'projections':
        download_projections(base)
    if args.source == 'vegas':
        download_vegas(base)
