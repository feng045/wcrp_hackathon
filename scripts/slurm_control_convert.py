from pathlib import Path
import subprocess as sp


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


print(sysrun('squeue -u mmuetz').stdout)
