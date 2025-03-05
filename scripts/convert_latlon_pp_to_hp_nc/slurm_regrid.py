import sys
from pathlib import Path
from convert_latlon_pp_to_hp_nc import UMRegridder

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

    regridder = UMRegridder('easygems_delaunay')

    for inoutpath in inout_paths[array_index * paths_per_job: (array_index + 1) * paths_per_job]:
        inpath = inoutpath[0]
        outpath_tpl = inoutpath[1]
        varname = inpath.parts[-2]

        tmp_outpath_tpl = outpath_tpl.parent / ('.regrid.tmp.' + outpath_tpl.name)
        print(varname, inpath, tmp_outpath_tpl)
        cfname = varname2cfname[varname]
        tmp_outpath_tpl.parent.mkdir(exist_ok=True, parents=True)

        regridder.run(inpath, tmp_outpath_tpl, cfname)

        for zoom in range(11)[::-1]:
            tmp_outpath = Path(str(tmp_outpath_tpl).format(zoom=zoom))
            outpath = Path(str(outpath_tpl).format(zoom=zoom))
            outpath.parent.mkdir(exist_ok=True, parents=True)
            print(tmp_outpath, outpath)
            tmp_outpath.rename(outpath)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
