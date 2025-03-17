import sys
import json
import math
import subprocess as sp
from collections import defaultdict
from pathlib import Path

import pandas as pd

from processing_config import processing_config


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


SLURM_SCRIPT = """#!/bin/bash
#SBATCH --job-name="RGlatlon2hp"
#SBATCH --time=04:00:00
#SBATCH --mem=64000
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --array=0-{njobs}
#SBATCH -o slurm/output/RGlatlon2hp_{date_string}_%a_%J.out
#SBATCH -e slurm/output/RGlatlon2hp_{date_string}_%a_%J.err

ARRAY_INDEX=${{SLURM_ARRAY_TASK_ID}}

python process_tasks.py slurm {tasks_path} ${{ARRAY_INDEX}} {paths_per_job}
"""


def parse_date_from_pp_path(path):
    return path.stem.split('.')[-1].split('_')[1]


def write_tasks_slurm_job(tasks):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    tasks_path = Path(f'slurm/tasks_{date_string}.json')
    # print(json.dumps(tasks, indent=4))

    with tasks_path.open('w') as f:
        json.dump(tasks, f, indent=4)

    slurm_script_path = Path(f'slurm/regrid_script_{date_string}.sh')
    paths_per_job = 1
    njobs = int(math.ceil(len(tasks) / paths_per_job))
    slurm_script_path.write_text(
        SLURM_SCRIPT.format(njobs=njobs, tasks_path=tasks_path, paths_per_job=paths_per_job,
                            date_string=date_string))
    return slurm_script_path


def main():
    config_key = sys.argv[1]
    config = processing_config[config_key]
    basedir = Path('/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3')
    donedir = Path('/gws/nopw/j04/hrcm/mmuetz/DYAMOND3_example_data/healpix')

    # Search for pp_paths with a specific date (N.B. filename sensitive).
    pp_paths = sorted(basedir.glob('glm/field.pp/apve*/**/*.pp'))
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(list)
    for path in pp_paths:
        dates_to_paths[parse_date_from_pp_path(path)].append(path)

    # Only keep completed downloads.
    dates_to_paths = {
        k: v for k, v in dates_to_paths.items()
        if len(v) == 4
    }

    # Build a list of tasks for all donepaths that don't exist.
    tasks = []
    for date in dates_to_paths:
        donepath = (donedir / '.slurm_done'/ f'DYAMOND3_example_data/sample_data_hierarchy/{config_key}/{date}.done')
        donepath.parent.mkdir(parents=True, exist_ok=True)
        if donepath.exists():
            print(f'{date}: already processed')
        else:
            if False and date == config['first_date']:
                # Be smart about running first task (which sets up .zarr store).
                init_tasks = [
                    {
                        'task_type': 'regrid',
                        'date': date,
                        'inpaths': [str(p) for p in dates_to_paths[date]],
                        'donepath': str(donepath),
                    }
                ]
                slurm_script_path = write_tasks_slurm_job(init_tasks)
                output = sysrun(f'sbatch {slurm_script_path}').stdout
                print(output)
            else:
                print(f'{date}: processing')
                tasks.append(
                    {
                        'task_type': 'regrid',
                        'date': date,
                        'inpaths': [str(p) for p in dates_to_paths[date]],
                        'donepath': str(donepath),
                    }
                )

    if len(tasks):
        slurm_script_path = write_tasks_slurm_job(tasks)
        print(slurm_script_path)
        print(sysrun(f'sbatch {slurm_script_path}').stdout)
    else:
        print('No tasks to run')


if __name__ == '__main__':
    main()
