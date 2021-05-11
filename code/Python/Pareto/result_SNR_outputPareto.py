import os,glob
from time import sleep
import numpy as np
# Select all 'graphml' files in current directory
myl=sorted(glob.glob('*graphml'))
print (myl)

# Weight Parameters latency
w1=0.0     #bandwidth
w2=1     #latency
w3=0.0    #cost

for j in (myl):
    for i in range(8,9,1):
    	for q in np.arange(0.04,0.5,0.04):
    		w1=q/2
    		w2=1-q
    		w3=q/2
    		os.system("python.exe SNR-Subset_Nodes-Custom-Cost-Delay.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, round(w1,2), round(w2,2), round(w3,2)))

# Weight Parameters latency
w1=1.0     #bandwidth
w2=0     #latency
w3=0.0    #cost

for j in (myl):
    for i in range(8,9,1):
    	for q in np.arange(0.04,0.5,0.04):
    		w1=1-q
    		w2=q/2
    		w3=q/2
    		os.system("python.exe SNR-Subset_Nodes-Custom-BW.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, round(w1,2), round(w2,2), round(w3,2)))
            

# Weight Parameters cost
w1=0.0     #bandwidth
w2=0     #latency
w3=1.0    #cost

for j in (myl):
    for i in range(8,9,1):
    	for q in np.arange(0.04,0.5,0.04):
    		w1=q/2
    		w2=q/2
    		w3=1-q
    		os.system("python.exe SNR-Subset_Nodes-Custom-Cost-Delay.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, round(w1,2), round(w2,2), round(w3,2)))
