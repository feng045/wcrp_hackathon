# coding: utf-8
import xarray as xr
xr.open_dataset('experimental_20200101T0000Z_pa000.hpz0-10.nc')
ds = xr.open_dataset('experimental_20200101T0000Z_pa000.hpz0-10.nc')
ds
import kerchunk.hdf
import fsspec
fs = fsspec.filesystem('file')
fs
fs.ls('*.nc')
get_ipython().run_line_magic('ls', '')
fs.ls('./*.nc')
fs.glob('./*.nc')
fs.glob('./*pa???.nc')
fs.glob('./*pa???.hpz0-10.nc')
fs.glob('./*pa???.hpz0-10.nc')
paths = fs.glob('./*pa???.hpz0-10.nc')
for p in paths:
    print(p)
    with fs.open(p, 'r') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        open(Path(p).with_suffix('.json'), 'wb').write(h5chunks.translate())
        
for p in paths:
    print(p)
    with fs.open(p, 'r') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        open(Path(p).with_suffix('.json'), 'wb').write(h5chunks.translate().encode())
        
for p in paths:
    print(p)
    with fs.open(p, 'r') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(h5chunks.translate())
        
for p in paths:
    print(p)
    with fs.open(p, 'r') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(h5chunks.translate().encode())
        
for p in paths:
    print(p)
    with fs.open(p, 'rb') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(h5chunks.translate().encode())
        
from pathlib import Path
for p in paths:
    print(p)
    with fs.open(p, 'rb') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(h5chunks.translate().encode())
        
for p in paths:
    print(p)
    with fs.open(p, 'rb') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(h5chunks.translate())
        
import simplejson
import ujson
ujson.dumps({'a': 1})
for p in paths:
    print(p)
    with fs.open(p, 'rb') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(ujson.dumps(h5chunks.translate()))
        
from kerchunk.combine import MultiZarrToZarr
mzz = MultiZarrToZarr(fs.glob('*.json'))
fs.glob('*.json')
mzz = MultiZarrToZarr(fs.glob('*.json'))
mzz = MultiZarrToZarr(fs.glob('*.json'), concat_dims=['time'])
mzz
d = mzz.translate()
d
Path('experimental_20200101T0000Z_pa.combined.json').write_text(d)
Path('experimental_20200101T0000Z_pa.combined.json').write_text(ujson.dumps(d))
fs2 = fsspec.filesystem('reference', fo='experimental_20200101T0000Z_pa.combined.json')
fs2
fs2.get_mapper('')
m = fs2.get_mapper('')
import xarray as xr
ds = xr.open_dataset(m, engine='zarr', backend_kwargs={'consolidated': False})
ds = xr.open_dataset(m, engine='zarr')
ds = xr.open_dataset(fs2, engine='zarr')
get_ipython().run_line_magic('pinfo', 'fsspec.filestystem')
get_ipython().run_line_magic('pinfo', 'fsspec.filesystem')
fs2 = fsspec.filesystem('.', fo='experimental_20200101T0000Z_pa.combined.json')
fs2 = fsspec.filesystem('file', fo='experimental_20200101T0000Z_pa.combined.json')
m = fs2.get_mapper('')
ds = xr.open_dataset(m, engine='zarr')
ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
                    "consolidated": False,
                    "storage_options": {"fo": 'experimental_20200101T0000Z_pa.combined.json'}
                    })
ds
ds.time
ds.toa_outgoing_longwave_flux_10.nbytes
ds.toa_outgoing_longwave_flux_10.nbytes / 1e9
ds.toa_outgoing_longwave_flux_1.nbytes / 1e9
ds.toa_outgoing_longwave_flux_1.nbytes
da = ds.toa_outgoing_longwave_flux_1.data
da
type(da)
da.shape
ds.chunks
ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
                    "consolidated": False,
                    "storage_options": {"fo": 'experimental_20200101T0000Z_pa.combined.json'}
                    })
ds.toa_outgoing_longwave_flux_2.sum()
ds.toa_outgoing_longwave_flux_10.sum()
ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
                    "consolidated": False,
                    "storage_options": {"fo": 'experimental_20200101T0000Z_pa.combined.json'},
                    'lazy': True,
                    })
ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
                    "consolidated": False,
                    "storage_options": {"fo": 'experimental_20200101T0000Z_pa.combined.json', 'lazy': True},
                                        })
ds
ds.cell_0.data
ds.toa_outgoing_longwave_flux_1.data
type(ds.toa_outgoing_longwave_flux_1.data)
type(ds.toa_outgoing_longwave_flux_10.data)
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr')
ds2 = xr.open_dataset('experimental_20200101T0000Z_pa.combined.zarr/')
ds2 = xr.open_dataset('experimental_20200101T0000Z_pa.combined.zarr/', engine='zarr')
ds2
ds2.chunks
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': (6, 12582912 / 12))
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': (6, 12582912 / 12)})
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'cunks': (6, 12582912 / 12)}})
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'cunks': (6, 12582912 / 12)}})
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'chunks': (6, 12582912 / 12)}})
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'chunks': (6, 12582912 // 12)}})
ds.to_zarr('experimental_20200101T0000Z_pa.combined.zarr', encoding={'toa_outgoing_longwave_flux_10': {'chunks': (6, 12582912 // 12)}})
ds2 = xr.open_dataset('experimental_20200101T0000Z_pa.combined.zarr/', engine='zarr')
ds2
ds2.toa_outgoing_longwave_flux_10.data
paths = fs.glob('./*pa???.hpz0-10.nc') + fs.glob('./*pa????.hpz0-10.nc')
for p in paths:
    print(p)
    with fs.open(p, 'rb') as fp:
        h5chunks = kerchunk.hdf.SingleHdf5ToZarr(fp, p)
        Path(p).with_suffix('.json').write_text(ujson.dumps(h5chunks.translate()))
        
