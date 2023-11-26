# beat-the-streak

This repository contains a test bed for evaluating different models for the MLB Beat the Streak contest.  

# The data

Different models in this repository may use different granularities of data.   All data is derived from pitch-level statcast data, obtained from [pybaseball](https://github.com/jldbc/pybaseball).  

### Obtaining the data 

First, you must set an environmetn variable that says where the data will live.  If you are on Ubuntu or OS X, add this line to your .bashrc file.

```$BTSDATA=/path/to/bts/data```

If you are on Windows, go to System --> Settings --> Advanced --> Environment Variables and add an envionrment variable "BTSDATA" that points to the directory where you want the data to live.  

1. **The Manual Way:**

The first step is to "pull", or download the data.  Navigate to ```src/bts/data``` and run

```
$ python pull.py --source retrosheet --start 2010-01-01 --end 2010-12-31
$ python pull.py --source statcast --start 2010-01-01 --end 2010-12-31
```

This will download all statcast and retrosheet data for 2010 and dump it to your ```$BTSDATA``` directory.  This will take a while, so it's best to let it run in the background while you do something else.  

The data currently exists as a set of csv files, one per date.  The second step is to load these in, clean and aggregate them, an dump them back out to the ```$BTSDATA``` directory.

```
$ python process.py
```

should do the trick.  After doing these two steps, if you want to load data in the future, you can use:

```python
>>> from bts.data.load import load_data, load_atbats

>>> data = load_data()  # batter/game data
>>> atbtas = load_atbats()  # atbat data
```

### Understanding the data

There are three levels of granularity in the data, 

1. The highest-level data contains information about (batter, game) pairs, that includes a 'hit' column specifying whether the given batter earned at least one hit in the given game, as well as other metadata relevant to the game (batter team, pitching team, ballpark, home, starting pitcher, etc.)

2. The next level of data contains information about each at bat, including the outcome of the at bat, and information about the at bat (batter, pitcher, other relevant context, etc.)

3. The finest granularity of data contains information about every pitch thrown, including the outcome of the pitch and relevant context.  


### Feature engineering

Some may be surprised that the data does not directly contain statisical information, such as the batting average of a given batter, the BA against the opposing pitcher, etc.  If desired, these statistics can easily be calculated from at-bat level data, however.  Since the statistics needed may vary by model, they should be computed within each model.  

# The models

Ultimately we are interested in the coarsest-grained data; i.e., will a batter earn a hit in a given game.  The finer granularity data may be useful for making predictions about this, however.  Models for this task should live in the bts/models/player_game folder, and they must implement three key functions

- **train**: trains a model using historical data
- **update**: upate a model using new data (typically, data for one day)
- **predict**: predict the probability of a hit for each (batter,game) in the input data

# Evaluation

To evaluate the model, we simulate the model running over 1 seasons worth of data.  As a concrete example, suppose we wish to evaluate our model on the 2019 season.  We begin by training our model using 2010-2018 data.  Then, one day at a time, we will predict the hit probability for every (batter, game) on the given day, and record the predicted probability along with the observed outcome (hit or no hit).  Then the data for that day is used to update the model, and the next day is predicted.  This process is repeated until the end of the regular season.  At the end of this process, the predictions and analyzed and statistics are reported together with helpful figures to understand model performance.  We can evaluate a model by navigating to the experiments folder and running

```python main.py --model baseline --year 2019```

### Notes

Unlike in traditional ML settings, this evaluation is not a standard one time train/test split.  Instead, we use multiple train/test split, where the test set corresponds to one day's worth of data, and the training data is all data that preceded it.  This more faithfully represents the sequential nature of the problem . Since data from earlier in the same season can be highly relevant for predicting outcomes later in the season, it is important to include it in the training dataset.  


# Results


### (player, game) model

Results are available at this [Google Sheet](https://docs.google.com/spreadsheets/d/10TmsPNHuWNbhQiL_0QaIlKTaAfK8i5i0O-XNKUFv23Y/edit?usp=sharing).


### atbat model

Below we show the (geometric) average likelihood of different models using different test sets.  

| Model | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.5783 | 0.5783 | 0.5826 | 0.5801 | 0.5850 | 0.5849 |
| Elo (Notebook) | 0.5849 | 0.5852 | 0.5896 | 0.5870 | 0.5928 | 0.5932 |
| Logistic | 0.5852 | 0.5856 | 0.5900 | 0.5872 | 0.5929 | 0.5938 |
| NeuralNet | - | - | - | 0.5857 | - | - |
