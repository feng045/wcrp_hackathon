# coding: utf-8
import numpy as np
a = np.arange(48)
a
a[22] = np.nan
a = np.arange(48).astype(float)
a[22] = np.nan
a
a.mean()
np.nanmean(a)
a.reshape(-1, 4)
np.nanmean(a.reshape(-1, 4), axis=1)
np.nanmean(np.nanmean(a.reshape(-1, 4), axis=1))
a = np.arange(8).astype(float)
np.nanmean(np.nanmean(a.reshape(-1, 4), axis=1))
a[2] = np.nan
np.nanmean(np.nanmean(a.reshape(-1, 4), axis=1))
np.nanmean(a)
a
np.nanmean(a.reshape(-1, 4), axis=1)
get_ipython().run_line_magic('pinfo', 'np.nanmean')
a
np.nanmean(a.reshape(-1, 4), axis=1)
aa = a.reshape(-1, 4)
aa
np.isnan(aa)
np.isnan(aa).sum(axis=-1)
np.isnan(aa).sum(axis=-1) / 4
w = np.isnan(aa).sum(axis=-1) / 4
w
aa
np.nanmean(aa, axis=-1)
np.average(np.nanmean(aa, axis=-1), w)
np.average(np.nanmean(aa, axis=-1), weights=w)
np.nanmean(aa)
aa
np.nansum(aa, axis=-1)
np.nansum(aa, axis=-1) * w
w
np.nansum(aa, axis=-1) * (1 - w)
np.average(np.nanmean(aa, axis=-1), weights=1-w)
a = np.arange(48).astype(float)
aa = a.reshape(-1, 4)
np.nanmean(aa)
w = np.isnan(aa).sum(axis=-1) / 4
np.average(np.nanmean(aa, axis=-1), weights=1-w)
a = np.arange(8).astype(float)
a[2] = np.nan
a = np.arange(8).astype(float)
a[6] = np.nan
np.nanmean(a)
22 / 7
41 / 12
np.nanmean(a.reshape(-1, 4), axis=1)
np.nanmean(a.reshape(-1, 4), axis=1).mean()
w = np.isnan(aa).sum(axis=-1) / 4
aa = a.reshape(-1, 4)
np.average(np.nanmean(aa, axis=-1), weights=1-w)
w = np.isnan(aa).sum(axis=-1) / 4
np.average(np.nanmean(aa, axis=-1), weights=1-w)
1-w
aa
np.nanmean(aa, axis=-1)
a = np.arange(12 * 4**2).astype(float)
a
a[6] = np.nan
aa = a.reshape(-1, 4)
np.nanmean(a)
w = 1 - np.isnan(aa).sum(axis=-1) / 4
np.average(np.nanmean(aa, axis=-1), weights=w)
aa = np.nanmean(a.reshape(-1, 4), axis=-1)
aa
aa.shape
w
np.average(aa, weights=w)
np.nanmean(a)
aaa = np.nanmean(aa.reshape(-1, 4), axis=-1)
aaa
ww = w.reshape(-1, 4).mean(axis=-1)
ww
np.average(aaa, weights=ww)
w = 1 - np.isnan(np.reshape(-1, 4)).sum(axis=-1) / 4
w = 1 - np.isnan(a.reshape(-1, 4)).sum(axis=-1) / 4
w
s = np.isnan(a.reshape(-1, 4)).sum(axis=-1)
s
ss = s.reshape(-1, 4).sum(axis=-1)
ss
np.average(aaa, weights=1 - ss/16)
np.average(aa, weights=w)
aaa
ww
aa
aaa = np.nanmean(aa.reshape(-1, 4) * ww, axis=-1)
aaa = np.nanmean(aa.reshape(-1, 4) * w, axis=-1)
aaa = np.nanmean((aa * w).reshape(-1, 4), axis=-1)
ww = w.reshape(-1, 4).mean(axis=-1)
np.average(aa, weights=w)
np.average(np.nanmean(aa, axis=-1), w)
np.average(aa, weights=w)
np.average(aaa, weights=ww)
np.nanmean(a)
aaa = np.nanmean((aa * w).reshape(-1, 4), axis=-1)
np.average(aaa, weights=ww)
aaa = (aa * w).reshape(-1, 4).sum(axis=-1) / w.sum()
np.average(aaa, weights=ww)
aaa = (aa * w).reshape(-1, 4).sum(axis=-1) / w.reshape(-1, 4).sum(axis=-1)
np.average(aaa, weights=ww)
get_ipython().run_line_magic('save', 'coarsen.ipython.py 1-108')
