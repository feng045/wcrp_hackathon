import zarr

root = zarr.group('zroot.zarr')
foo = root.create_group(name="foo")

bar = root.create_array(
    name="bar", shape=(100, 10), chunks=(10, 10), dtype="f4"
)
spam = foo.create_array(name="spam", shape=(10,), dtype="i4")

# Assign values
bar[:, :] = np.random.random((100, 10))
