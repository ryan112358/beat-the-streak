import glob
import pandas as pd
from bts.evaluation import metrics
from main import process_results

def get_all_results():
    results = {}
    for item in glob.glob('*/results.csv'):
        df = pd.read_csv(item)
        folder = item.split('/')[0]
        process_results(df, folder, stdout=False)
        name, year = folder[:-5], folder[-4:]
        for metric, score_fn in metrics.ALL_METRICS.items():
            results[(name, year, metric)] = score_fn(df)
        print(f'Recomputed metrics + visualizations for {folder}')
    return pd.Series(results)


if __name__ == '__main__':
    results = get_all_results()
    results.to_clipboard()
    print('Saved aggregate results to clipboard.  Ready to paste into Google Sheets.')
