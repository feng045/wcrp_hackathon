# coding: utf-8
import sys
import json
from pathlib import Path
import subprocess as sp

import pandas as pd


# jobids_path = Path(sys.argv[1])
# with jobids_path.open('r') as f:
#     jobids = json.load(f)
# jobids_str = ','.join(str(j) for j in jobids)
jobids_str = sys.argv[1]


cmd = rf"sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' -j {jobids_str}|grep -E '^[0-9_]*\.batch\|' --color=Never"
# print(cmd)

result = sp.run(cmd, capture_output=True, text=True, shell=True)
# print(result.stdout)
lines = [l for l in result.stdout.split('\n') if l]
data = [line.split('|') for line in lines]

df = pd.DataFrame(data, columns=["jobid", "start", "end", "elapsed", "state", "maxrss", "host"])
df = df[df.state != 'FAILED']
df['start'] = pd.to_datetime(df['start'], errors='coerce')
df['end'] = pd.to_datetime(df['end'], errors='coerce')
df['maxrss'] = df['maxrss'].str.rstrip('K').replace('', '0').astype(float) / 1e6
print(df.to_string())

df["elapsed"] = pd.to_timedelta(df["elapsed"])

# df.state == 'COMPLETED'
# df[df.state == 'COMPLETED'].elapsed

print('mean:', df[df.state == 'COMPLETED'][['elapsed', 'maxrss']].mean())
print('min:', df[df.state == 'COMPLETED'][['elapsed', 'maxrss']].min())
print('max:', df[df.state == 'COMPLETED'][['elapsed', 'maxrss']].max())
print('earliest start:', df[df.state == 'COMPLETED'].start.min())
print('latest start:', df[df.state == 'COMPLETED'].end.max())
