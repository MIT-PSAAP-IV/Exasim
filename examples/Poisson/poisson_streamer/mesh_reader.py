from pathlib import Path
import numpy as np

def read_msh3(filename):
    lines=[line.strip() for line in Path(filename).read_text().splitlines()]
    start=lines.index("$Nodes"); count=int(lines[start+1]); points=np.empty((2,count))
    for i in range(count): points[:,i]=np.asarray(lines[start+2+i].split(),dtype=float)[1:3]
    start=lines.index("$Elements"); count=int(lines[start+1]); cells=[]
    for i in range(count):
        record=np.asarray(lines[start+2+i].split(),dtype=np.int64)
        if record[1]==2: cells.append(record[4:7]-1)
    return points,np.asarray(cells,dtype=np.int64).T
