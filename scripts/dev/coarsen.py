import numpy as np

a = np.arange(12 * 4**2).astype(float)
a[6] = np.nan

aa = np.nanmean(a.reshape(-1, 4), axis=-1)
w = 1 - np.isnan(a.reshape(-1, 4)).sum(axis=-1) / 4

aaa = (aa * w).reshape(-1, 4).sum(axis=-1) / w.reshape(-1, 4).sum(axis=-1)
ww = w.reshape(-1, 4).mean(axis=-1)

print(np.nanmean(a))
print(np.average(aa, weights=w))
print(np.average(aaa, weights=ww))

def hp_coarsen_with_weights(field, weights=None):
    if weights is None:
        coarse_field = np.nanmean(field.reshape(-1, 4), axis=-1)
        new_weights = 1 - np.isnan(field.reshape(-1, 4)).sum(axis=-1) / 4
    else:
        coarse_field = (field * weights).reshape(-1, 4).sum(axis=-1) / weights.reshape(-1, 4).sum(axis=-1)
        new_weights = weights.reshape(-1, 4).mean(axis=-1)
    return coarse_field, new_weights

aa, w = hp_coarsen_with_weights(a)
aaa, ww = hp_coarsen_with_weights(aa, w)

print()
print(np.nanmean(a))
print(np.average(aa, weights=w))
print(np.average(aaa, weights=ww))

z = 10
a = np.arange(12 * 4**z).astype(float)
# a[6] = np.nan
r = np.random.randint(0, len(a), 100000)
r.sort()
a[r] = np.nan

new_field = a
nf2 = new_field.copy()
new_weights = None

print()
print(np.nanmean(a))
for i in range(11)[::-1]:
    new_field, new_weights = hp_coarsen_with_weights(new_field, new_weights)
    print(i, np.sum(new_field * new_weights) / new_weights.sum())

print()
print(np.nanmean(a))
for i in range(11)[::-1]:
    nf2 = np.nanmean(nf2.reshape(-1, 4), axis=-1)
    print(i, np.nanmean(nf2))

