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

def download_statcast(start, end, base):
    for day in pd.date_range(start, end):
        daystr = day.strftime('%Y-%m-%d')
        print(daystr)
        data = statcast(daystr)

        if not os.path.isdir(base + '/' + daystr):
            os.makedirs(base + '/' + daystr)

        data.to_csv('%s/%s/statcast.csv' % (base, daystr), index=False)

def download_retrosheet(start, end, base):
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

    return params

if __name__ == '__main__':

    description = 'download data and store in appropriate folder'
    formatter = argparse.ArgumentDefaultsHelpFormatter    
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--start', help='start date (yyyy-mm-dd)')
    parser.add_argument('--end', help='end date (yyyy-mm-dd)')

    base = os.environ['BTSDATA']

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    #download_statcast(start, end, base)
    download_retrosheet(start, end, base)
    #download_weather(start, end, base)
