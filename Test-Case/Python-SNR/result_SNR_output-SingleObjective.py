import os,glob
from time import sleep
import numpy

# Select all 'graphml' files in current directory
myl=sorted(glob.glob('*graphml'))
print (myl)


#----------------------------Pareto front for bandwidth------------------------
# Weight Parameters latency
#w1=0.0     #bandwidth
#w2=1     #latency
#w3=0.0    #cost
for j in (myl):
    for i in range(8,9,1):
        for q in numpy.arange(0,1.0,0.1):
            w1=round((1-q),2)
            w2=round(q/2,2)
            w3=round(q/2,2)
            os.system("python.exe SNR_singleObjective.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
                  %(j,i,w1, w2, w3, j.replace(".graphml", ""),i, w1, w2, w3))
#            
#            
#----------------------------Pareto front for delay----------------------------
# Weight Parameters latency
w1=0.0     #bandwidth
w2=1     #latency
w3=0.0    #cost
for j in (myl):
    for i in range(8,9,1):
    	for q in numpy.arange(0,1.0,0.1):
    		w1=round(q/2,2)
    		w2=round((1-q),2)
    		w3=round(q/2,2)
    		os.system("python.exe SNR_singleObjective.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, w1, w2, w3))


##----------------------------Pareto front for cost----------------------------
## Weight Parameters cost
#w1=0.0     #bandwidth
#w2=0     #latency
#w3=1.0    #cost       
for j in (myl):
    for i in range(8,9,1):
        for q in numpy.arange(0,1.0,0.1):
            w1=round(q/2,2)
            w2=round(q/2,2)
            w3=round((1-q),2)
            os.system("python.exe SNR_singleObjective.py %s %d %s %s %s > %s-%d-BW%sLatency%sCost%s.txt"
    			%(j,i,w1, w2, w3, j.replace(".graphml", ""),i, w1, w2, w3))