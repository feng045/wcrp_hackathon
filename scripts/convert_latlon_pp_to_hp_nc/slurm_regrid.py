import sys

sys.path.insert(0, '/home/users/mmuetz/deploy/global_hackathon_tools/dataset_transforms')
from pathlib import Path
from um_latlon_pp_to_healpix_nc import UMRegridder


def main(input_output_files, array_index, paths_per_job=10):
    outdir = Path('/gws/nopw/j04/hrcm/mmuetz/DYAMOND3_example_data/healpix')

    print(input_output_files, array_index)
    lines = Path(input_output_files).read_text().split('\n')
    indone_paths = []
    for line in lines:
        indone_paths.append([Path(f) for f in line.split(',')])

    regridder = UMRegridder('easygems_delaunay')

    for inpath, donepath in indone_paths[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        outpath_tpl = outdir / f'{{varname}}/hpz{{zoom}}/{inpath.stem}.{{varname}}.hpz{{zoom}}.nc'

        regridder.run(inpath, outpath_tpl)
        donepath = Path(donepath)
        donepath.write_text(f'Converted from {inpath} to {outpath_tpl}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
