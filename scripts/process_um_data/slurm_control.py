import json
import subprocess as sp
import sys
from collections import defaultdict
from itertools import batched
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from calc_completed_chunks import find_tgt_calcs, find_tgt_time_calcs
from processing_config import processing_config

SLURM_SCRIPT_ARRAY = """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --time=10:00:00
#SBATCH --mem={mem}
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --array=0-{njobs}%{nconcurrent_tasks}
#SBATCH -o slurm/output/{job_name}_{config_key}_{date_string}_%A_%a.out
#SBATCH -e slurm/output/{job_name}_{config_key}_{date_string}_%A_%a.err
#SBATCH --comment={comment}
# These nodes repeatedly fail to be able to read the kscale GWS.
#SBATCH --exclude=host1012,host1077,host1106
{dependency}

# Quick check to see if it can access the kscale GWS.
if ! ls /gws/nopw/j04/kscale > /dev/null 2>&1; then
    echo "ERROR: kscale GWS not accessible on $(hostname)! Exiting."
    exit 99
fi

ARRAY_INDEX=${{SLURM_ARRAY_TASK_ID}}

python um_process_tasks.py slurm {tasks_path} ${{ARRAY_INDEX}}
"""


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


def sbatch(slurm_script_path):
    try:
        return sysrun(f'sbatch --parsable {slurm_script_path}').stdout.strip()
    except sp.CalledProcessError as e:
        logger.error(f'sbatch failed with exit code {e.returncode}')
        logger.error(e)
        raise


def parse_date_from_pp_path(path):
    datestr = path.stem.split('.')[-1].split('_')[1]
    return pd.to_datetime(datestr, format="%Y%m%dT%H")


def write_tasks_slurm_job_array(config_key, tasks, job_name, nconcurrent_tasks=30, depends_on=None, mem=100000):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    tasks_path = Path(f'slurm/tasks/tasks_{job_name}_{config_key}_{date_string}.json')
    logger.debug(tasks_path)
    logger.trace(json.dumps(tasks, indent=4))

    if depends_on:
        dependency = f'#SBATCH --dependency=afterok:{depends_on}'
    else:
        dependency = ''

    with tasks_path.open('w') as f:
        json.dump(tasks, f, indent=4)

    comment = f'{config_key},{job_name}'

    slurm_script_path = Path(f'slurm/scripts/script_{job_name}_{config_key}_{date_string}.sh')
    njobs = len(tasks) - 1
    slurm_script_path.write_text(
        SLURM_SCRIPT_ARRAY.format(job_name=job_name, config_key=config_key, njobs=njobs,
                                  mem=mem,
                                  nconcurrent_tasks=nconcurrent_tasks,
                                  tasks_path=tasks_path,
                                  dependency=dependency,
                                  date_string=date_string,
                                  comment=comment))
    return slurm_script_path


def find_dyamond3_pp_dates_to_paths(basedir):
    # Search for pp_paths with a specific date (N.B. filename sensitive).
    pp_paths = sorted(basedir.glob('field.pp/apve*/**/*.pp'))
    logger.debug(f'found {len(pp_paths)} pp paths')
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(list)
    for path in pp_paths:
        # These appear after about 2020-02-20 - not sure why.
        # Not sure what's in them either.
        if 'apvere' in path.stem:
            continue
        dates_to_paths[parse_date_from_pp_path(path)].append(path)
    # Only keep completed downloads.
    dates_to_paths = {
        k: v for k, v in dates_to_paths.items()
        if len(v) == 4
    }
    logger.debug(f'found {len(dates_to_paths)} complete dates')
    return dates_to_paths


def write_jobids(jobids):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    jobids_path = Path(f'slurm/jobids/jobids_{date_string}.json')
    with jobids_path.open('w') as f:
        json.dump(jobids, f, indent=4)
    logger.info(f'written jobids to: {jobids_path}')


@click.group()
@click.option('--dry-run', '-n', is_flag=True)
@click.option('--debug', '-D', is_flag=True)
@click.option('--trace', '-T', is_flag=True)
@click.option('--nconcurrent-tasks', '-N', default=40, type=int)
@click.pass_context
def cli(ctx, dry_run, debug, trace, nconcurrent_tasks):
    ctx.ensure_object(dict)
    ctx.obj['dry_run'] = dry_run
    ctx.obj['nconcurrent_tasks'] = nconcurrent_tasks
    logger.remove()
    if trace:
        logger.add(sys.stderr, level="TRACE")
    elif debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    if dry_run:
        logger.warning("Dry run: not launching any jobs")

    for path in ['slurm/tasks', 'slurm/scripts', 'slurm/output', 'slurm/jobids/']:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


@cli.command()
@click.argument('config_key')
@click.pass_context
def process(ctx, config_key):
    nconcurrent_tasks = ctx.obj['nconcurrent_tasks']

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

    dates_to_paths = find_dyamond3_pp_dates_to_paths(basedir)

    create_jobid = None
    jobids = []

    # Build a list of tasks for all donepaths that don't exist.
    tasks = []
    for date in dates_to_paths:
        # if not (date.year == 2020 and date.month == 1):
        #     continue
        if date == config['first_date']:
            create_donepath = donedir / donepath_tpl.format(task='create_empty_zarr_store', date=date)
            logger.debug(create_donepath)
            if not create_donepath.exists():
                logger.info('Creating zarr store')
                create_task = {
                    'task_type': 'create_empty_zarr_stores',
                    'config_key': config_key,
                    'date': str(date),
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(create_donepath),
                }
                slurm_script_path = write_tasks_slurm_job_array(config_key, [create_task], f'createzarr',
                                                                nconcurrent_tasks=nconcurrent_tasks)
                logger.debug(slurm_script_path)
                create_donepath.parent.mkdir(parents=True, exist_ok=True)
                if not ctx.obj['dry_run']:
                    create_jobid = sbatch(slurm_script_path)
                    logger.info(f'create empty zarr stores jobid: {create_jobid}')
                    jobids.append(create_jobid)

        donepath = (donedir / donepath_tpl.format(task='regrid', date=date))
        donepath.parent.mkdir(parents=True, exist_ok=True)
        if donepath.exists():
            logger.debug(f'{date}: already processed')
        else:
            logger.info(f'{date}: processing')
            tasks.append(
                {
                    'task_type': 'regrid',
                    'config_key': config_key,
                    'date': str(date),
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(donepath),
                }
            )

    regrid_jobid = None
    if len(tasks):
        logger.info(f'Running {len(tasks)} tasks')
        slurm_script_path = write_tasks_slurm_job_array(config_key, tasks, 'regrid',
                                                        nconcurrent_tasks=nconcurrent_tasks,
                                                        depends_on=create_jobid)
        logger.debug(slurm_script_path)
        if not ctx.obj['dry_run']:
            regrid_jobid = sbatch(slurm_script_path)
        logger.debug(f'regrid jobid: {regrid_jobid}')
        jobids.append(regrid_jobid)
    else:
        logger.info('No tasks to run')

    if not ctx.obj['dry_run']:
        write_jobids(jobids)


@cli.command()
@click.argument('config_key')
@click.pass_context
def coarsen(ctx, config_key):
    # This needs to be it's own command. The reason is that I need to be able to examine a completed
    # zarr store to be able to work out how to divvy up the work. This can only been done once the above have completed,
    # and can't be calculated in advance. It would be possible to have a job whose sole purpose was to do this calc and
    # then launch other jobs, but this seems overly complicated. Just wait until these have completed then launch.
    # coarsen_task = {
    #     'task_type': 'coarsen_healpix_region',
    #     'config_key': config_key,
    # }
    # slurm_script_path = write_tasks_slurm_job_array(config_key, [coarsen_task], 'coarsen', nconcurrent_tasks=nconcurrent_tasks,
    #                                                 depends_on=regrid_jobid)
    # logger.debug(slurm_script_path)
    # coarsen_jobid = sysrun(f'sbatch --parsable {slurm_script_path}').stdout.strip()
    # jobids.append(coarsen_jobid)
    nconcurrent_tasks = ctx.obj['nconcurrent_tasks']
    config = processing_config[config_key]
    freqs = {
        '2d': 'PT1H',
        '3d': 'PT3H',
    }
    # Last variables for each.
    if config_key != 'glm.n1280_GAL9_nest':
        variables = {
            '2d': 'rsutcs',
            '3d': 'qs',
        }
    else:
        # qs not in glm.n1280_GAL9_nest
        variables = {
            '2d': 'rsutcs',
            '3d': 'zg',
        }
    jobids = []
    for dim in ['3d', '2d']:
        prev_zoom_job_id = None

        rel_url_tpl = config['zarr_store_url_tpl'][5:]  # chop off 's3://'
        freq = freqs[dim]
        urls = {
            z: rel_url_tpl.format(freq=freq, zoom=z)
            for z in range(11)
        }

        chunks = config['groups'][dim]['chunks']
        max_zoom = config['max_zoom']
        variable = variables[dim]

        tgt_calcs = find_tgt_calcs(urls, chunks=chunks, variable=variable, max_zoom=max_zoom, dim=dim)
        tgt_time_calcs = find_tgt_time_calcs(tgt_calcs, chunks=chunks)

        for zoom in range(max_zoom - 1, -1, -1):
            if zoom not in tgt_time_calcs:
                continue
            tasks = []
            for tgt_times in batched(tgt_time_calcs[zoom], 10):
                tasks.append(
                    {
                        'task_type': 'coarsen',
                        'config_key': config_key,
                        'tgt_zoom': zoom,
                        'dim': dim,
                        'tgt_times': tgt_times,
                    }
                )
            if len(tasks):
                logger.info(f'Running {len(tasks)} tasks')
                if dim == '3d':
                    mem = 256000
                else:
                    mem = 100000
                slurm_script_path = write_tasks_slurm_job_array(config_key, tasks, f'coarsen_{dim}_{zoom}',
                                                                mem=mem,
                                                                nconcurrent_tasks=nconcurrent_tasks,
                                                                depends_on=prev_zoom_job_id)
                logger.debug(slurm_script_path)

                if not ctx.obj['dry_run']:
                    prev_zoom_job_id = sbatch(slurm_script_path)
                jobids.append(prev_zoom_job_id)

    if not ctx.obj['dry_run']:
        write_jobids(jobids)


@cli.command()
@click.pass_context
def ls(ctx):
    for key in processing_config:
        print(key)


if __name__ == '__main__':
    cli(obj={})
