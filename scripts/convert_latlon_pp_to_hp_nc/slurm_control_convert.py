import json
import subprocess as sp
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

from processing_config import processing_config


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


SLURM_SCRIPT_ARRAY = """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --time=10:00:00
#SBATCH --mem=100000
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --array=0-{njobs}%{nconcurrent_tasks}
#SBATCH -o slurm/output/{job_name}_{config_key}_{date_string}_%J_%a.out
#SBATCH -e slurm/output/{job_name}_{config_key}_{date_string}_%J_%a.err
{dependency}

ARRAY_INDEX=${{SLURM_ARRAY_TASK_ID}}

python um_process_tasks.py slurm {tasks_path} ${{ARRAY_INDEX}}
"""


def parse_date_from_pp_path(path):
    return path.stem.split('.')[-1].split('_')[1]


def write_tasks_slurm_job_array(config_key, tasks, job_name, nconcurrent_tasks=30, depends_on=None):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    tasks_path = Path(f'slurm/{job_name}_tasks_{date_string}.json')
    logger.debug(tasks_path)
    logger.trace(json.dumps(tasks, indent=4))

    if depends_on:
        dependency = f'#SBATCH --dependency=afterok:{depends_on}'
    else:
        dependency = ''

    with tasks_path.open('w') as f:
        json.dump(tasks, f, indent=4)

    slurm_script_path = Path(f'slurm/script_{job_name}_{config_key}_{date_string}.sh')
    njobs = len(tasks) - 1
    slurm_script_path.write_text(
        SLURM_SCRIPT_ARRAY.format(job_name=job_name, config_key=config_key, njobs=njobs,
                                  nconcurrent_tasks=nconcurrent_tasks,
                                  tasks_path=tasks_path,
                                  dependency=dependency,
                                  date_string=date_string))
    return slurm_script_path


def main_array():
    nconcurrent_tasks = 40

    config_key = sys.argv[1]
    logger.debug(f'using {nconcurrent_tasks} concurrent tasks')
    logger.info(f'Running for {config_key}')
    config = processing_config[config_key]
    logger.trace(config)
    basedir = config['basedir']
    donedir = config['donedir']
    donepath_tpl = config['donepath_tpl']
    logger.debug(f'basedir: {basedir}')
    logger.debug(f'donedir: {donedir}')
    logger.debug(f'donepath_tpl: {donepath_tpl}')

    # Search for pp_paths with a specific date (N.B. filename sensitive).
    pp_paths = sorted(basedir.glob('field.pp/apve*/**/*.pp'))
    logger.debug(f'found {len(pp_paths)} pp paths')
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(list)
    for path in pp_paths:
        dates_to_paths[parse_date_from_pp_path(path)].append(path)

    # Only keep completed downloads.
    dates_to_paths = {
        k: v for k, v in dates_to_paths.items()
        if len(v) == 4
    }
    logger.debug(f'found {len(dates_to_paths)} complete dates')

    create_jobid = None

    # Build a list of tasks for all donepaths that don't exist.
    tasks = []
    for date in dates_to_paths:
        if not (date.startswith('202001') or date.startswith('202002')):
            continue
        if date == config['first_date']:
            create_donepath = donedir / donepath_tpl.format(task='create_empty_zarr_store', date=date)
            logger.debug(create_donepath)
            if not create_donepath.exists():
                logger.info('creating zarr store')
                create_task = {
                    'task_type': 'create_empty_zarr_stores',
                    'config_key': config_key,
                    'date': date,
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(create_donepath),
                }
                slurm_script_path = write_tasks_slurm_job_array(config_key, [create_task], 'createzarr',
                                                                nconcurrent_tasks=nconcurrent_tasks)
                logger.debug(slurm_script_path)
                create_donepath.parent.mkdir(parents=True, exist_ok=True)
                create_jobid = sysrun(f'sbatch --parsable {slurm_script_path}').stdout
                logger.info(f'create empty zarr stores jobid: {create_jobid}')

        donepath = (donedir / donepath_tpl.format(task='regrid', date=date))
        donepath.parent.mkdir(parents=True, exist_ok=True)
        if donepath.exists():
            logger.info(f'{date}: already processed')
        else:
            logger.info(f'{date}: processing')
            tasks.append(
                {
                    'task_type': 'regrid',
                    'config_key': config_key,
                    'date': date,
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(donepath),
                }
            )

    regrid_jobid = None
    if len(tasks):
        logger.info(f'Running {len(tasks)} tasks')
        slurm_script_path = write_tasks_slurm_job_array(config_key, tasks, 'regrid', nconcurrent_tasks=nconcurrent_tasks,
                                                        depends_on=create_jobid)
        logger.debug(slurm_script_path)
        regrid_jobid = sysrun(f'sbatch --parsable {slurm_script_path}').stdout
        logger.info(f'Regrid jobid: {regrid_jobid}')
    else:
        logger.info('No tasks to run')


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    main_array()
