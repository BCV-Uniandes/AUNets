import subprocess
from string import Template
import os
import time

start = time.time()
binaryPath = '/.../brox'
testImageL = '/.../frame0.png'
testImageR = '/.../frame0.png'

baseTargetImage = '/.../myFlowOutput'

myCommand = Template('$bnPath -left $lImage -right $rImage -ziel $zImage')

processes = set()
max_processes = 5

for i in range(0, 1000):
    actualCMD = myCommand = Template(
        '$bnPath -left $lImage -right $rImage -ziel $zImage').substitute(
            bnPath=binaryPath,
            lImage=testImageL,
            rImage=testImageR,
            zImage=baseTargetImage + str(i) + '.jpg')
    print(actualCMD, ' actualCMD')
    processes.add(subprocess.Popen(actualCMD, shell=True))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])
end = time.time()
print
print('Done, about ', (end - start))
