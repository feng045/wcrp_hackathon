from collections import defaultdict
from pathlib import Path

import pandas as pd

from processing_config import processing_config


# def main():


if __name__ == '__main__':
    # main()
    start_date = '2020-01-20'
    end_date = '2021-04-01'
    dates = pd.date_range(start=start_date, end=end_date, freq='12h')
    streams = 'abcd'
    sims = list(processing_config)
    exists_paths = defaultdict(list)
    for sim in sims:
        config = processing_config[sim]
        basedir = config['basedir']
        for stream in streams:
            for date in dates:
                # e.g. /gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200329T00.pp
                datestr = date.strftime('%Y%m%dT%H')
                path = basedir / f'field.pp/apver{stream}.pp/{sim}.apver{stream}_{datestr}.pp'
                if path.exists():
                    exists_paths[sim].append(path)

    frac_done = {
        k: len(v) / (len(dates) * len(streams))
        for k, v in exists_paths.items()
    }

    n = max(len(k) for k in frac_done.keys())
    for k, v in frac_done.items():
        print(f'{k:<{n}}: {v * 100:6.2f}%')

                
