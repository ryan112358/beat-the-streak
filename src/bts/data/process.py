import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
#from pybaseball.retrosheet import events
from pybaseball import playerid_reverse_lookup
from pybaseball import player_search_list
import os
import pickle

ROOT = os.environ['BTSDATA']

def process_all_data():
    pitches = load_statcast()
    atbats = atbats_from_pitches(pitches)
    games = games_from_atbats(atbats)
    info, order = load_retrosheet(pitches) 
    #weather = load_weather()

    pitches = pitches.merge(info, on=['game_pk'])
    atbats = atbats.merge(info, on=['game_pk'])
    games = games.merge(info, on=['game_pk'])
    games = games.merge(order, on=['game_pk', 'batter'])

    games['game_date'] = pd.to_datetime(games['game_date'])
    atbats['game_date'] = pd.to_datetime(atbats['game_date'])
    pitches['game_date'] = pd.to_datetime(pitches['game_date'])

    games['game_pk'] = games['game_pk'].astype('category')
    atbats['game_pk'] = atbats['game_pk'].astype('category')
    pitches['game_pk'] = pitches['game_pk'].astype('category')

    players = list(set(pitches.batter).union(set(pitches.pitcher)))
    A = playerid_reverse_lookup(players).set_index('key_mlbam')
    name_lookup = (A['name_first'] + ' ' + A['name_last']).str.title()

    games['batter'] = games.batter.map(name_lookup).astype('category')
    games['pitcher'] = games.pitcher.map(name_lookup).astype('category')
    atbats['batter'] = atbats.batter.map(name_lookup).astype('category')
    atbats['pitcher'] = atbats.pitcher.map(name_lookup).astype('category')
    pitches['batter'] = pitches.batter.map(name_lookup).astype('category')
    pitches['pitcher'] = pitches.pitcher.map(name_lookup).astype('category')
   
    lookup = {'SF': 90, 'LAA': 45, 'STL': 60, 'ARI': 0, 'NYM': 30, 'PHI': 15, 'DET': 150, 'COL': 0, 'LAD': 30, 'BOS': 45, 'CIN': 120, 'KC': 45, 'MIL': 135, 'HOU': 345, 'WSH': 30, 'OAK': 60, 'BAL': 30, 'SD': 0, 'PIT': 120, 'CLE': 0, 'TEX': 135, 'TOR': 0, 'SEA': 45, 'MIA': 75, 'MIN': 90, 'TB': 45, 'ATL': 30, 'CWS': 135, 'CHC': 30, 'NYY': 75}
    for col in ['ballpark', 'pitcher_team', 'batter_team']:
        games.loc[games[col]=='FLA', col] = 'MIA'
        atbats.loc[atbats[col]=='FLA', col] = 'MIA'
        pitches.loc[pitches[col]=='FLA', col] = 'MIA'
        games[col] = games[col].astype('str').astype('category')
        atbats[col] = atbats[col].astype('str').astype('category')
        pitches[col] = pitches[col].astype('str').astype('category')
    games['park_angle'] = games.ballpark.map(lookup)
    atbats['park_angle'] = atbats.ballpark.map(lookup)
    pitches['park_angle'] = pitches.ballpark.map(lookup)

    
    pickle.dump(games, open(ROOT + '/data.pkl', 'wb'))
    pickle.dump(atbats, open(ROOT + '/atbats.pkl', 'wb'))
    pickle.dump(pitches, open(ROOT + '/pitches.pkl', 'wb'))
     
    return pitches, atbats, games


def load_statcast():
    files = glob.glob(ROOT + '/*/')
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f + 'statcast.csv'))
        except:
            continue
    df = pd.concat(dfs)
    df['year'] = df['game_year']
    cols = ['game_pk', 'game_date', 'year', 'batter', 'pitcher', 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 'events', 'description', 'spin_dir', 'zone', 'stand', 'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y', 'sv_id', 'vx0', 'vy0', 'vz0', 'sz_top', 'sz_bot', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'if_fielding_alignment', 'of_fielding_alignment', 'spin_axis']
    df = df[df.game_type == 'R'][cols]
    df['game_pk'] = df['game_pk'].astype(str)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    df['spray_angle'] = np.arctan((df.hc_x - 125.42)/(198.27-df.hc_y))*180/np.pi
    df['at_bat_number'] = df.at_bat_number.astype(int)
    df['pitch_number'] = df.pitch_number.astype(int) 
    df['batter_team'] = df['home_team']
    df['pitcher_team'] = df['home_team']
    df.loc[df.inning_topbot=='Top', 'batter_team'] = df.loc[df.inning_topbot=='Top', 'away_team']
    df.loc[df.inning_topbot=='Bot', 'pitcher_team'] = df.loc[df.inning_topbot=='Bot', 'away_team']
    df['on_3b'] = df['on_3b'].notnull()
    df['on_2b'] = df['on_2b'].notnull()
    df['on_1b'] = df['on_1b'].notnull()
    df['home'] = df.inning_topbot == 'Bot'
    df['ballpark'] = df['home_team']
    del df['home_team']
    del df['away_team']
    df['year'] = df['year'].astype(int)

    return df.reset_index(drop=True).sort_values(by=['game_date', 'game_pk','at_bat_number','pitch_number'])

def load_retrosheet(pitches):
    info = []
    start = []
    curr_id = None

    for name in glob.glob(ROOT + '/retrosheet/*.EV*'):
        # Strips the newline character
        for line in open(name, 'r').readlines():
            line = line.strip()
            if line[:2] == 'id':
                curr_id = line[3:]
                info.append({ 'id' : curr_id })
            elif line[:4] == 'info':
                split = line.split(',')
                info[-1][split[1]] = split[2]
            elif line[:5] == 'start':
                _, code, name, home_flag, order, position = line.split(',')
                start.append({ 'id': curr_id, 'batter_id': code, 'batter_name': name.strip('"'), 'home': home_flag=='1', 'order': order, 'position': position })
    
    cols = ['id', 'visteam', 'hometeam', 'site', 'date', 'number', 'starttime', 'daynight', 'umphome', 'temp', 'winddir', 'windspeed', 'precip', 'sky']
                
    info = pd.DataFrame(info)[cols]
    info['hour'] = pd.to_datetime(info['starttime']).map(lambda x: x.hour)
    cols = ['site', 'daynight', 'umphome', 'winddir', 'precip', 'sky', 'starttime']
    info[cols] = info[cols].astype('category')
    info[['windspeed', 'temp']] = info[['windspeed', 'temp']].astype(float)
    start = pd.DataFrame(start)
    start['order'] = start.order.astype(int)
    start['position'] = start.position.astype(int)

    retrosheet_gameids = start[start.position == 1]
    retrosheet_gameids = retrosheet_gameids.set_index(['id','home']).batter_id.unstack('home').reset_index()
    retrosheet_gameids = retrosheet_gameids.rename(columns={False:'away_starter', True:'home_starter'})
    retrosheet_gameids[['home_team','game_date']] = retrosheet_gameids.id.str.split("2", n=1, expand=True)
    parse_date = lambda s: "%s-%s-%s" % (s[:4], s[4:6], s[6:8])
    retrosheet_gameids['game_date'] = ("2" + retrosheet_gameids.game_date.str[:-1]).map(parse_date)
    mapping = playerid_reverse_lookup(retrosheet_gameids.away_starter, key_type='retro').set_index('key_retro').key_mlbam
    retrosheet_gameids['away_starter'] = retrosheet_gameids.away_starter.map(mapping)
    mapping = playerid_reverse_lookup(retrosheet_gameids.home_starter, key_type='retro').set_index('key_retro').key_mlbam
    retrosheet_gameids['home_starter'] = retrosheet_gameids.home_starter.map(mapping)
    retrosheet_gameids = retrosheet_gameids.astype(str)
    #retrosheet_gameids['game_date'] = pd.to_datetime(retrosheet_gameids['game_date'])
    mapping = {'PIT': 'PIT', 'HOU': 'HOU', 'KC': 'KC', 'CWS': 'CWS', 'PHI': 'PHI', 'BOS': 'BOS', 'NYM': 'NYM', 'WSH': 'WSH', 'STL': 'STL', 'MIA': 'MIA', 'FLA': 'FLA', 'SEA': 'SEA', 'ARI': 'ARI', 'TOR': 'TOR', 'LAD': 'LAD', 'TB': 'TB', 'MIL': 'MIL', 'ATL': 'ATL', 'CIN': 'CIN', 'MIN': 'MIN', 'CHC': 'CHC', 'TEX': 'TEX', 'CLE': 'CLE', 'NYY': 'NYY', 'OAK': 'OAK', 'DET': 'DET', 'LAA': 'LAA', 'BAL': 'BAL', 'SF': 'SF', 'COL': 'COL', 'SD': 'SD', 'WAS': 'WSH', 'TBA': 'TB', 'SLN': 'STL', 'SFN': 'SF', 'SDN': 'SD', 'NYN': 'NYM', 'NYA': 'NYY', 'LAN': 'LAD', 'KCA': 'KC', 'FLO': 'FLA', 'CHN': 'CHC', 'CHA': 'CWS', 'ANA': 'LAA'}
    retrosheet_gameids['ballpark'] = retrosheet_gameids.home_team.map(mapping)

    df = pitches
    first_atbat = df.groupby(['game_date','ballpark','home'], observed=True).at_bat_number.min().to_frame().reset_index()
    first_atbat['pitch_number'] = 1
    first_atbat = first_atbat.merge(df, how='left', on=['game_date','ballpark','home','at_bat_number','pitch_number'])
    statcast_gameids = first_atbat[['game_date','ballpark','home','pitcher','game_pk']].sort_values(['game_date', 'ballpark']).copy()
    statcast_gameids = statcast_gameids.astype(str).set_index(['game_date','ballpark','game_pk','home']).pitcher.unstack().reset_index()
    statcast_gameids = statcast_gameids.rename(columns={"True":'away_starter', "False":'home_starter'})

    tmp = statcast_gameids.merge(retrosheet_gameids, on=['game_date', 'ballpark', 'home_starter'])
    retrosheet_to_statcast = tmp.set_index('id').game_pk    

    order = start[start.order != 0].copy()
    order['game_pk'] = retrosheet_to_statcast[order['id']].values
    lookup = playerid_reverse_lookup(start.batter_id, key_type='retro').set_index('key_retro')['key_mlbam']
    order['batter'] = lookup[order['batter_id']].values
    order = order[['game_pk', 'batter', 'order']]

    info['game_pk'] = retrosheet_to_statcast[info['id']].values
    info = info[['game_pk', 'site', 'starttime', 'daynight', 'umphome', 'temp', 'winddir', 'windspeed', 'precip', 'sky']]
    #info['game_date'] = pd.to_datetime(info['game_date'])

    return info, order

def load_weather():
    weather = pd.read_csv(ROOT + '/weather.zip')
    weather['game_date'] = weather.time.str[:10]
    weather['time'] = weather.time.str[11:]
    weather['hour'] = weather.time.str[:2].astype(int)
    weather['temperature'] = weather.temp*9/5 + 32
    weather = weather.drop(columns=['snow', 'tsun', 'wpgt', 'temp', 'time'])
    weather['coco'] = weather.coco.fillna(0)
    weather = weather.interpolate()
    weather.loc[weather.ballpark=='WAS', 'ballpark'] = 'WSH'
    return weather

def atbats_from_pitches(pitches):
    events = ['field_out', 'strikeout', 'single', 'walk', 'double', 'home_run', 'force_out', 'grounded_into_double_play', 'hit_by_pitch', 'field_error', 'sac_fly', 'sac_bunt', 'triple', 'intent_walk', 'double_play', 'fielders_choice', 'fielders_choice_out', 'strikeout_double_play', 'other_out', 'catcher_interf', 'sac_fly_double_play', 'wild_pitch', 'triple_play', 'sac_bunt_double_play']
    cols = ['game_pk', 'game_date', 'year', 'batter_team', 'pitcher_team', 'batter', 'pitcher', 'pitch_type', 'events', 'stand', 'p_throws', 'ballpark', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'at_bat_number', 'if_fielding_alignment', 'of_fielding_alignment', 'home']
    atbats = pitches[pitches.events.isin(events)][cols]
    atbats['hit'] = atbats.events.isin(['single','double','triple','home_run'])
    return atbats

def games_from_atbats(atbats):
    cols = ['game_pk', 'game_date', 'year', 'batter_team', 'pitcher_team', 'ballpark', 'home', 'pitcher', 'stand', 'p_throws']
    games = atbats.groupby(['game_pk', 'batter_team'], observed=True).head(1).reset_index()[cols]
    hits = atbats.groupby(['game_pk', 'batter_team', 'batter'],observed=True).hit.any().reset_index()
    return hits.merge(games, on=['game_pk', 'batter_team'])

if __name__ == '__main__':
    pitches, atbats, games = process_all_data()

    from IPython import embed; embed()
