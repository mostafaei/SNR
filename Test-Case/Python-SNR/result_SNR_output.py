import os,glob
from time import sleep

# Select all 'graphml' files in current directory
myl=sorted(glob.glob('*graphml'))
print (myl)

# Weight Parameters
w1=0.05     #bandwidth
w2=0.99     #latency
w3=0.05     #cost

for j in (myl):
    for i in range(8,10,1):
        os.system("python.exe SNR-Subset_Nodes-Custom.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
                  %(j,i,w1, w2, w3, j.replace(".graphml", ""),i, w1, w2, w3))

