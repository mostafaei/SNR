import os,glob
from time import sleep

# Select all 'graphml' files in current directory
myl=sorted(glob.glob('*graphml'))
print (myl)

# Weight Parameters latency
w1=0.0     #bandwidth
w2=1     #latency
w3=0.0    #cost

for j in (myl):
    for i in range(8,9,1):
    	for q in numpy.arange(0.05,5,0.05):
    		w1=q
    		w2=1-q
    		w3=q
    		os.system("python.exe SNR-Subset_Nodes-Custom.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, w1, w2, w3))
