import sys
from pathlib import Path
# from convert_latlon_pp_to_hp_nc import convert_latlon_pp_to_healpix_nc
from convert_latlon_pp_to_hp_nc import convert_latlon_cube_to_healpix


varname2cfname = {
    'pe_T': 'air_temperature',
    'OLR': 'toa_outgoing_longwave_flux',
}

def main(input_output_files, array_index, paths_per_job=10):
    print(input_output_files, array_index)
    lines = Path(input_output_files).read_text().split('\n')
    inout_paths = []
    for line in lines:
        inout_paths.append([Path(f) for f in line.split(',')])

    for inoutpath in inout_paths[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        inpath = inoutpath[0]
        outpaths = inoutpath[1:]
        varname = inpath.parts[-2]

        tmp_outpaths = [o.parent / ('.regrid.tmp.' + o.name) for o in outpaths]
        print(varname, inpath, tmp_outpaths, outpaths)
        cfname = varname2cfname[varname]
        tmp_outpaths[0].parent.mkdir(exist_ok=True, parents=True)

        convert_latlon_cube_to_healpix(inpath, tmp_outpaths, cfname)
        for tmp_outpath, outpath in zip(tmp_outpaths, outpaths):
            outpath.parent.mkdir(exist_ok=True, parents=True)
            print(tmp_outpath, outpath)
            tmp_outpath.rename(outpath)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
