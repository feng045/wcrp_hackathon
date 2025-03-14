import json
import math
import subprocess as sp
from collections import defaultdict
from pathlib import Path

import pandas as pd


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

python slurm_regrid.py {input_output_files} ${{ARRAY_INDEX}} {paths_per_job}
"""


def parse_date_from_pp_path(path):
    return path.stem.split('.')[-1].split('_')[1]

def main():
    # basedir = Path('/gws/nopw/j04/hrcm/hackathon/')
    basedir = Path('/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3')
    outdir = Path('/gws/nopw/j04/hrcm/mmuetz/DYAMOND3_example_data/healpix')

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

    todo_dates_to_paths = []
    donepaths = []
    for date in dates_to_paths:
        donepath = (outdir / '.slurm_done'/ f'DYAMOND3_example_data/sample_data_hierarchy/5km-RAL3/{date}.done')
        donepath.parent.mkdir(parents=True, exist_ok=True)
        donepaths.append(donepath)
        if not donepath.exists():
            todo_dates_to_paths.append((date, [str(p) for p in dates_to_paths[date]], str(donepath)))

    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    input_output_files = Path(f'slurm/input_output_files_{date_string}.json')
    todo_dates_to_paths = todo_dates_to_paths[:2]
    print(json.dumps(todo_dates_to_paths, indent=4))
    with input_output_files.open('w') as f:
        json.dump(todo_dates_to_paths, f, indent=4)

    slurm_script_path = Path(f'slurm/regrid_script_{date_string}.sh')
    print(slurm_script_path)
    paths_per_job = 1
    njobs = int(math.ceil(len(todo_dates_to_paths) / paths_per_job))

    slurm_script_path.write_text(
        SLURM_SCRIPT.format(njobs=njobs, input_output_files=input_output_files, paths_per_job=paths_per_job,
                            date_string=date_string, ))

    print(sysrun(f'sbatch {slurm_script_path}').stdout)


def main_old():
    # basedir = Path('/gws/nopw/j04/hrcm/hackathon/')
    basedir = Path('/gws/nopw/j04/kscale/DYAMOND3_example_data/sample_data_hirerarchy/5km-RAL3')
    outdir = Path('/gws/nopw/j04/hrcm/mmuetz/DYAMOND3_example_data/healpix')

    pp_paths = sorted(basedir.glob('**/*.pp'))
    pp_paths = [p for p in pp_paths if p.is_file()]
    # pp_paths = [p for p in pp_paths if ('OLR' in str(p)) or ('pe_T' in str(p))]
    donepaths = []
    for pp_path in pp_paths:
        donepath = (outdir / '.slurm_done').joinpath(*pp_path.parts[1:]).with_suffix('.done')
        donepath.parent.mkdir(parents=True, exist_ok=True)
        donepaths.append(donepath)

    lines = []
    for inpath, donepath in zip(pp_paths, donepaths):
        if not donepath.exists():
            lines.append(','.join([str(inpath), str(donepath)]))

    if not len(lines):
        print('No files to convert.')
        return

    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    input_output_files = Path(f'slurm/input_output_files_{date_string}.txt')
    input_output_files.write_text('\n'.join(lines))
    print(input_output_files)

    slurm_script_path = Path(f'slurm/regrid_script_{date_string}.sh')
    print(slurm_script_path)

    # paths_per_job = 5
    # njobs = 5
    paths_per_job = 20
    njobs = int(math.ceil(len(lines) / paths_per_job))

    slurm_script_path.write_text(
    SLURM_SCRIPT.format(njobs=njobs, input_output_files=input_output_files, paths_per_job=paths_per_job,
        date_string=date_string, ))

    print(sysrun(f'sbatch {slurm_script_path}').stdout)


if __name__ == '__main__':
    main()
