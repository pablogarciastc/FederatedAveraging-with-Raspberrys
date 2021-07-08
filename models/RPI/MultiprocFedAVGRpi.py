#!/usr/bin/env python3

from multiprocessing import Process
import os
import sys

script_path = 'FedAVG.py'

def run(script, name,client, num_epochs, nsuscriber):
    os.system("{} -i {} {} {} {}".format(script, name,client,num_epochs,nsuscriber))  

if __name__ == '__main__':
    script_path = './FedAVG.py'
    nrpi = int(sys.argv[1])
    nprocesos = int(sys.argv[2])
    i = 1    
    for client in range(((nrpi - 1) * nprocesos + 1) , ((nrpi - 1) * nprocesos + 1 + nprocesos)):
        p = Process(target=run, args=(script_path, client,sys.argv[1],sys.argv[3],i))
        p.start()
        i = i + 1
    p.join()


