"""Aggregate, clean, and dump baseball data."""
import pandas as pd
import glob
import numpy as np

from pybaseball import playerid_reverse_lookup
from pybaseball import player_search_list
import os
import pickle
import time
from IPython import embed

ROOT = os.environ["BTSDATA"]
# Used for converting retrosheet --> statcast as well as
# merging statcast ballparks into a canonical representation.
BALLPARK_MAPPING = {
    "ANA": "LAA",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHA": "CWS",
    "CHC": "CHC",
    "CHN": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "CWS": "CWS",
    "DET": "DET",
    "FLA": "MIA",
    "FLO": "MIA",
    "HOU": "HOU",
    "KC": "KC",
    "KCA": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "LAN": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "MON": "WSH",
    "NYA": "NYY",
    "NYM": "NYM",
    "NYN": "NYM",
    "NYY": "NYY",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SDN": "SD",
    "SEA": "SEA",
    "SF": "SF",
    "SFN": "SF",
    "SLN": "STL",
    "STL": "STL",
    "TB": "TB",
    "TBA": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WAS": "WSH",
    "WSH": "WSH",
}

# Only relevant for weather data, which is currently unused
PARK_ANGLES = {
    "ARI": 0,
    "ATL": 30,
    "BAL": 30,
    "BOS": 45,
    "CHC": 30,
    "CIN": 120,
    "CLE": 0,
    "COL": 0,
    "CWS": 135,
    "DET": 150,
    "HOU": 345,
    "KC": 45,
    "LAA": 45,
    "LAD": 30,
    "MIA": 75,
    "MIL": 135,
    "MIN": 90,
    "NYM": 30,
    "NYY": 75,
    "OAK": 60,
    "PHI": 15,
    "PIT": 120,
    "SD": 0,
    "SEA": 45,
    "SF": 90,
    "STL": 60,
    "TB": 45,
    "TEX": 135,
    "TOR": 0,
    "WSH": 30,
}


def process_all_data(pattern="*"):
    t0 = time.time()
    # First we will load and lightly augment the pitches data, before further processing
    pitches = load_statcast(pattern)

    print('Loaded and cleaned statcast data', time.time() - t0)

    # Now augment with player names.
    # Note: it is very important that pitches data and retrosheet data is
    # consistent before merged w.r.t. mlbid/player name.
    player_lookup = player_lookup_from_statcast(pitches)
    pitches["batter"] = pitches.batter.map(player_lookup).astype("category")
    pitches["pitcher"] = pitches.pitcher.map(player_lookup).astype("category")

    # First make the retrosheet data joinable with the statcast data
    # We will not perform the join here, since retrosheet is not available
    # in real time.  For now, models that need this data have to load it
    # and perform the join themselves.
    info, starters = load_retrosheet()


    print('Loaded and cleaned retrosheet data', time.time() - t0)

    retrosheet_gameids = get_retrosheet_gameids(info, starters)
    retrosheet_gameids["home_starter"] = retrosheet_gameids.home_starter.map(
        player_lookup
    )
    
    print('Computed retrosheet game_ids', time.time() - t0)

    # If pattern != * some nulls are expected here
    # assert retrosheet_gameids.home_starter.notnull().all()
    statcast_gameids = get_statcast_gameids(pitches)
    joinable_retrosheet = make_retrosheet_joinable(
        info, statcast_gameids, retrosheet_gameids
    )

    print('Computed statcast game_ids and make joinable retrosheet', time.time() - t0)
    

    # Now get coarser grained views of the data
    atbats = atbats_from_pitches(pitches)
    batter_games = games_from_atbats(atbats)

    # Now we will add lineup order information to the batter/games data
    statcast_lineups = lineups_from_statcast(pitches)
    batter_games = batter_games.merge(
        statcast_lineups, how="left", on=["game_pk", "batter"]
    )
    batter_games = batter_games[batter_games.order.notnull()]
    
    print('Derived atbats and batter/game data from pitch data', time.time() - t0)

    return pitches, atbats, batter_games, joinable_retrosheet


def load_statcast(pattern="*"):

    cols = {
        "at_bat_number": int,
        "away_score": int,
        "away_team": "category",
        "ax": float,
        "ay": float,
        "az": float,
        "babip_value": float,
        "balls": int,
        "bat_score": int,
        "batter": "category",
        "bb_type": "category",
        # "delta_home_win_exp": float,
        "delta_run_exp": float,
        # "des": str,
        "description": "category",
        "effective_speed": float,
        "estimated_ba_using_speedangle": float,
        "estimated_woba_using_speedangle": float,
        "events": "category",
        # "fielder_2": int,
        # "fielder_2.1": int,
        # "fielder_3": int,
        # "fielder_4": int,
        # "fielder_5": int,
        # "fielder_6": int,
        # "fielder_7": int,
        # "fielder_8": int,
        # "fielder_9": int,
        "fld_score": int,
        "game_date": "datetime64[ns]",
        "game_pk": str,
        "game_type": str,
        "game_year": int,
        "hc_x": float,
        "hc_y": float,
        "hit_distance_sc": float,
        "hit_location": float,
        "home_score": int,
        "home_team": "category",
        "if_fielding_alignment": "category",
        "inning": int,
        "inning_topbot": str,
        "iso_value": float,
        "launch_angle": float,
        "launch_speed": float,
        "launch_speed_angle": float,
        "of_fielding_alignment": "category",
        "on_1b": str,
        "on_2b": str,
        "on_3b": str,
        "outs_when_up": int,
        "p_throws": "category",
        "pfx_x": float,
        "pfx_z": float,
        # "pitch_name": "category",
        "pitch_number": int,
        "pitch_type": "category",
        "pitcher": "category",
        # "pitcher.1": "object",
        "plate_x": float,
        "plate_z": float,
        # "player_name": str,
        # "post_away_score": int,
        # "post_bat_score": int,
        # "post_fld_score": int,
        # "post_home_score": int,
        "release_extension": float,
        "release_pos_x": float,
        "release_pos_y": float,
        "release_pos_z": float,
        "release_speed": float,
        "release_spin_rate": float,
        "spin_axis": float,
        "spin_dir": float,
        # "spin_rate_deprecated": float,
        "stand": "category",
        "strikes": int,
        # "sv_id": "object",
        "sz_bot": float,
        "sz_top": float,
        # "tfs_deprecated": float,
        # "tfs_zulu_deprecated": float,
        "type": "category",
        # "umpire": float,
        "vx0": float,
        "vy0": float,
        "vz0": float,
        # "woba_denom": float,
        # "woba_value": float,
        "zone": "category",
    }

    files = glob.glob(f"{ROOT}/{pattern}/")
    dfs = []
    for f in files:
        try:
            # We only load in the columns we need for space efficiency
            dfs.append(pd.read_csv(f + "statcast.csv", usecols=list(cols)))
        except:
            # Some files do not have the header, which causes exceptions.
            continue
    df = pd.concat(dfs).astype(cols)
    df = df[df["game_type"] == "R"]

    # Now apply some light cleanup
    df["spray_angle"] = np.arctan((df.hc_x - 125.42) / (198.27 - df.hc_y)) * 180 / np.pi

    # Derive batter/pitcher team from inning_topbot.
    df["home"] = df.inning_topbot == "Bot"
    df["batter_team"] = df["home_team"]
    df["pitcher_team"] = df["home_team"]
    df.loc[~df.home, "batter_team"] = df.loc[~df.home, "away_team"]
    df.loc[df.home, "pitcher_team"] = df.loc[df.home, "away_team"]

    # We don't care whos on base, just whether someone is or not.
    df["on_3b"] = df["on_3b"].notnull()
    df["on_2b"] = df["on_2b"].notnull()
    df["on_1b"] = df["on_1b"].notnull()
    df["ballpark"] = df["home_team"]

    # Finally sort the data by date/time and reset the index
    return df.sort_values(
        by=["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)


def load_retrosheet():
    info = []
    start = []
    curr_id = None

    # Load the data in from raw text files
    for name in glob.glob(ROOT + "/retrosheet/*.EV*"):
        # Strips the newline character
        for line in open(name, "r").readlines():
            line = line.strip()
            if line[:2] == "id":
                curr_id = line[3:]
                info.append({"id": curr_id})
            elif line[:4] == "info":
                split = line.split(",")
                info[-1][split[1]] = split[2]
            elif line[:5] == "start":
                _, code, name, home_flag, order, position = line.split(",")
                # Start contains information about the lineup order, which we now
                # retrieve directly from statcast.  However, it is still used
                # to derive the retrosheet_gameids
                start.append(
                    {
                        "id": curr_id,
                        "batter_id": code,
                        "batter_name": name.strip('"'),
                        "home": home_flag == "1",
                        "order": order,
                        "position": position,
                    }
                )
    # Now do some light cleanup

    info_cols = {
        "id": str,
        "visteam": str,
        "hometeam": str,
        "site": "category",
        "date": "datetime64[ns]",
        "number": int,
        "starttime": str,
        "daynight": "category",
        "umphome": "category",
        "temp": float,
        "winddir": "category",
        "windspeed": float,
        "precip": "category",
        "sky": "category",
    }
    start_cols = {
        "id": str,
        "batter_id": str,
        "batter_name": str,
        "home": bool,
        "order": int,
        "position": int,
    }

    info = pd.DataFrame(info)[list(info_cols)].astype(info_cols)
    start = pd.DataFrame(start).astype(start_cols)

    info["hour"] = pd.to_datetime(info["starttime"], format="%H:%M%p").map(
        lambda x: x.hour
    )
    return info, start


def get_retrosheet_gameids(info, start):

    retrosheet_gameids = start[start.position == 1]
    retrosheet_gameids = (
        retrosheet_gameids.set_index(["id", "home"])
        .batter_id.unstack("home")
        .reset_index()
    )
    retrosheet_gameids = retrosheet_gameids.rename(
        columns={False: "away_starter", True: "home_starter"}
    )
    retrosheet_gameids[["home_team", "game_date"]] = retrosheet_gameids.id.str.split(
        "2", n=1, expand=True
    )
    parse_date = lambda s: "%s-%s-%s" % (s[:4], s[4:6], s[6:8])
    retrosheet_gameids["game_date"] = ("2" + retrosheet_gameids.game_date.str[:-1]).map(
        parse_date
    )
    mapping = (
        playerid_reverse_lookup(retrosheet_gameids.away_starter, key_type="retro")
        .set_index("key_retro")
        .key_mlbam
    )
    retrosheet_gameids["away_starter"] = retrosheet_gameids.away_starter.map(mapping)
    mapping = (
        playerid_reverse_lookup(retrosheet_gameids.home_starter, key_type="retro")
        .set_index("key_retro")
        .key_mlbam
    )
    retrosheet_gameids["home_starter"] = retrosheet_gameids.home_starter.map(mapping)
    retrosheet_gameids["ballpark"] = retrosheet_gameids.home_team.map(BALLPARK_MAPPING)

    return retrosheet_gameids


def get_statcast_gameids(pitches):
    first_atbat = (
        pitches.groupby(["game_date", "ballpark", "home"], observed=True)
        .at_bat_number.min()
        .to_frame()
        .reset_index()
    )
    first_atbat["pitch_number"] = 1
    first_atbat = first_atbat.merge(
        pitches,
        how="left",
        on=["game_date", "ballpark", "home", "at_bat_number", "pitch_number"],
    )
    statcast_gameids = (
        first_atbat[["game_date", "ballpark", "home", "pitcher", "game_pk"]]
        .sort_values(["game_date", "ballpark"])
        .copy()
    )
    statcast_gameids = (
        statcast_gameids.astype(str)
        .set_index(["game_date", "ballpark", "game_pk", "home"])
        .pitcher.unstack()
        .reset_index()
    )
    statcast_gameids = statcast_gameids.rename(
        columns={"True": "away_starter", "False": "home_starter"}
    )
    return statcast_gameids


def make_retrosheet_joinable(retrosheet_data, statcast_gameids, retrosheet_gameids):
    # NOTE: for double headers from rain delay, statcast and retrosheet disagree about "home team"
    # Statcast says it's the team that bats in the bottom half
    # Retrosheet determines it based on the ballpark
    tmp = statcast_gameids.merge(
        retrosheet_gameids, how="left", on=["game_date", "ballpark", "home_starter"]
    )
    retrosheet_to_statcast = tmp.set_index("id").game_pk

    retrosheet_data = pd.merge(
        retrosheet_data, retrosheet_to_statcast, left_on="id", right_index=True
    )
    cols = [
        "game_pk",
        "site",
        "starttime",
        "daynight",
        "umphome",
        "temp",
        "winddir",
        "windspeed",
        "precip",
        "sky",
    ]

    return retrosheet_data[cols]


def load_weather():
    # Not currently used, currently getting weather data from retrosheet
    # Revisit in the future, since this allows you to get recent weather
    # and (I think) realtime predictions.
    weather = pd.read_csv(ROOT + "/weather.zip")
    weather["game_date"] = weather.time.str[:10]
    weather["time"] = weather.time.str[11:]
    weather["hour"] = weather.time.str[:2].astype(int)
    weather["temperature"] = weather.temp * 9 / 5 + 32
    weather = weather.drop(columns=["snow", "tsun", "wpgt", "temp", "time"])
    weather["coco"] = weather.coco.fillna(0)
    weather = weather.interpolate()
    weather.loc[weather.ballpark == "WAS", "ballpark"] = "WSH"
    return weather


def atbats_from_pitches(pitches):
    events = [
        "field_out",
        "strikeout",
        "single",
        "walk",
        "double",
        "home_run",
        "force_out",
        "grounded_into_double_play",
        "hit_by_pitch",
        "field_error",
        "sac_fly",
        "sac_bunt",
        "triple",
        "intent_walk",
        "double_play",
        "fielders_choice",
        "fielders_choice_out",
        "strikeout_double_play",
        "other_out",
        "catcher_interf",
        "sac_fly_double_play",
        "wild_pitch",
        "triple_play",
        "sac_bunt_double_play",
    ]
    cols = [
        "game_pk",
        "game_date",
        "game_year",
        "batter_team",
        "pitcher_team",
        "batter",
        "pitcher",
        "pitch_type",
        "events",
        "stand",
        "p_throws",
        "ballpark",
        "on_3b",
        "on_2b",
        "on_1b",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "if_fielding_alignment",
        "of_fielding_alignment",
        "home",
    ]
    atbats = pitches[pitches.events.isin(events)][cols]
    atbats["hit"] = atbats.events.isin(["single", "double", "triple", "home_run"])
    return atbats


def games_from_atbats(atbats):
    cols = [
        "game_pk",
        "game_date",
        "game_year",
        "batter_team",
        "pitcher_team",
        "ballpark",
        "home",
        "pitcher",
        "stand",
        "p_throws",
    ]
    games = (
        atbats.groupby(["game_pk", "batter_team"], observed=True)
        .head(1)
        .reset_index()[cols]
    )
    hits = (
        atbats.groupby(["game_pk", "batter_team", "batter"], observed=True)
        .hit.sum()
        .reset_index()
    )
    return hits.merge(games, on=["game_pk", "batter_team"])


def lineups_from_statcast(pitches):
    def compute_lineup(game_team):
        df = game_team.sort_values("at_bat_number").batter.drop_duplicates()
        # There is at least one dataframe with only 5 values.
        num = min(9, df.shape[0])
        if num < 9:
            # This was observed to happen in a spring training game.
            # Culprit was NaN batters (probably due to invalid player_id mapping)
            cols = ['game_date', 'game_pk', 'home_team', 'away_team']
            print('Found funny game', num, game_team.iloc[0].loc[cols])
        result = pd.DataFrame(index=np.arange(num) + 1)
        result["batter"] = df.iloc[:num].values
        result.index.name = "order"
        return result

    return (
        pitches.groupby(["game_pk", "batter_team"], observed=True)
        .apply(compute_lineup)
        .reset_index()
        .drop(columns="batter_team")
    )


def player_lookup_from_statcast(pitches):
    players = list(set(pitches.batter).union(set(pitches.pitcher)))
    A = playerid_reverse_lookup(players).set_index("key_mlbam")
    return (A["name_first"] + " " + A["name_last"]).str.title()


if __name__ == "__main__":
    pitches, atbats, batter_games, retrosheet = process_all_data()

    pitches.to_parquet(path=f"{ROOT}/pitches.parquet.gzip", compression="gzip")
    atbats.to_parquet(path=f"{ROOT}/atbats.parquet.gzip", compression="gzip")
    batter_games.to_parquet(path=f"{ROOT}/data.parquet.gzip", compression="gzip")
    retrosheet.to_parquet(path=f"{ROOT}/retrosheet.parquet.gzip", compression="gzip")

    # embed()
