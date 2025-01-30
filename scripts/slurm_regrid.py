import sys
from pathlib import Path
from convert_latlon_pp_to_hp_nc import convert_latlon_pp_to_healpix_nc


def main(input_output_files, array_index, paths_per_job=10):
    print(input_output_files, array_index)
    lines = Path(input_output_files).read_text().split('\n')
    inout_paths = []
    for line in lines:
        inout_paths.append(Path(f) for f in line.split(','))

    for inpath, outpath in inout_paths[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        varname = inpath.parts[-2]
        tmp_outpath = outpath.parent / ('.regrid.tmp.' + outpath.stem)
        print(varname, inpath, tmp_outpath, outpath)
        tmp_outpath.parent.mkdir(exist_ok=True, parents=True)
        convert_latlon_pp_to_healpix_nc(inpath, tmp_outpath, varname)
        tmp_outpath.rename(outpath)



if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
