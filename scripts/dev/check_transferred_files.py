from collections import defaultdict
from pathlib import Path

import pandas as pd

from processing_config import processing_config


# def main():

def scan_directories(dates):
    streams = 'abcd'
    sims = list(processing_config)
    data = {
        'sim': [],
        'stream': [],
        'date': [],
        'path': [],
        'exists': [],
        'size': [],
    }
    for sim in sims:
        config = processing_config[sim]
        basedir = config['basedir']
        for stream in streams:
            for date in dates:
                # e.g. /gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200329T00.pp
                datestr = date.strftime('%Y%m%dT%H')
                path = basedir / f'field.pp/apver{stream}.pp/{sim}.apver{stream}_{datestr}.pp'
                data['sim'].append(sim)
                data['stream'].append(stream)
                data['date'].append(date)
                data['path'].append(str(path))
                if path.exists():
                    # exists_paths[sim].append(path)
                    data['exists'].append(True)
                    data['size'].append(path.stat().st_size)
                else:
                    data['exists'].append(False)
                    data['size'].append(0)

    return pd.DataFrame(data)


if __name__ == '__main__':
    start_date = '2020-01-20'
    end_date = '2021-04-01'
    dates = pd.date_range(start=start_date, end=end_date, freq='12h')
    try:
        # i.e. Ipython: run -i script.py.
        df
    except NameError:
        df = scan_directories(dates)
    df_exists = df[df['exists']]

    curr_size = df_exists["size"].sum() / (2**40)

    print('Percent files transferred')
    print(df_exists.groupby(['sim'])['exists'].sum() / (len(dates) * 4) * 100)
    print()
    print('Percent transferred by size')
    print(df_exists.groupby(['sim'])['size'].sum() / (df_exists.groupby(['sim', 'stream'])['size'].mean().groupby('sim').sum() * len(dates)) * 100)
    print()
    print('Total estimated sizes for each sim')
    print(df_exists.groupby(['sim', 'stream'])['size'].mean().groupby('sim').sum() * len(dates) / (2**40))
    print()
    print(f'Current size: {curr_size:.3f}T')
    total_est_size = (df_exists.groupby(['sim', 'stream'])['size'].mean() * len(dates) / (2**40)).sum()
    print(f'Total estimated size: {total_est_size:.3f}T')
    print(f'Percent done: {curr_size / total_est_size * 100:.2f}%')
