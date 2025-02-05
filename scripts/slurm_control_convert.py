from pathlib import Path
import math
import subprocess as sp

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


if __name__ == '__main__':
    # print(sysrun('squeue -u mmuetz').stdout)
    basedir = Path('/gws/nopw/j04/hrcm/cache/torau/Lorenzo_u-cu087')
    outdir = Path('/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087')
    pp_paths = sorted(basedir.glob('**/*.pp'))
    pp_paths = [p for p in pp_paths if ('OLR' in str(p)) or ('pe_T' in str(p))]
    # print(pp_paths)
    outpaths = []
    for pp_path in pp_paths:
        varname = pp_path.parts[-2]
        outpath = Path(f'/gws/nopw/j04/hrcm/mmuetz/Lorenzo_u-cu087/{varname}/healpix') / ('experimental_' + pp_path.stem + '.hpz0-10.nc')
        outpaths.append(outpath)

    lines = []
    for inpath, outpath in zip(pp_paths, outpaths):
        if not outpath.exists():
            lines.append(f'{inpath},{outpath}')

    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    input_output_files = Path(f'slurm/input_output_files_{date_string}.txt')
    input_output_files.write_text('\n'.join(lines))

    slurm_script_path = Path(f'slurm/regrid_script_{date_string}.sh')

    # paths_per_job = 5
    # njobs = 5
    paths_per_job = 20
    njobs = int(math.ceil(len(lines) / paths_per_job))

    slurm_script_path.write_text(
        SLURM_SCRIPT.format(
            njobs=njobs,
            input_output_files=input_output_files,
            paths_per_job=paths_per_job,
            date_string=date_string,
        )
    )

    if len(lines):
        print(sysrun(f'sbatch {slurm_script_path}').stdout)


