import os
from string import Template
import subprocess
from shutil import copyfile

binaryPath = './broxDir'
processes = set()
max_processes = 20


def issueFlowCommandToPool(inputDir, outputDir):
    actualCMD = Template('$bnPath --source $sDir --ziel $zDir ').substitute(
        bnPath=binaryPath, sDir=inputDir, zDir=outputDir)
    print('Issuing to pool', actualCMD)
    processes.add(subprocess.Popen(actualCMD, shell=True))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])

    return 0  # No idea we need this?


def fillFinalImageInDir(path):
    print('Fix dir ', path)
    num_files = len(
        [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    try:
        copyfile(path + '/' + str(num_files - 1).zfill(4) + '.jpg',
                 path + '/' + str(num_files + 1).zfill(4) + '.jpg')
    except BaseException:
        try:
            copyfile(path + '/' + str(num_files - 1).zfill(3) + '.jpg',
                     path + '/' + str(num_files + 1).zfill(3) + '.jpg')
        except BaseException:
            try:
                copyfile(path + '/' + str(num_files - 1).zfill(0) + '.jpg',
                         path + '/' + str(num_files + 1).zfill(0) + '.jpg')
            except BaseException:
                print('Ignore')


def calculateOFBroxGPUAnSet(sourceDir, targetDir):
    # print('calculateOFBroxGPUAnSet')
    # print('sourceDir ',sourceDir)
    # print('targetDir ',targetDir)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        print('created ', targetDir)

    # if os.path.isfile(targetDir + '/0001.jpg'):#frames are there already
    # print ('Frames for  ', targetDir, ' OF already calculated')
    # fillFinalImageInDir(targetDir)
    # return
    numFilesSource = len([
        f for f in os.listdir(sourceDir)
        if os.path.isfile(os.path.join(sourceDir, f))
    ])
    numFilesTarget = len([
        f for f in os.listdir(targetDir)
        if os.path.isfile(os.path.join(targetDir, f))
    ])

    if numFilesSource != numFilesTarget:
        print('numFilesSource', numFilesSource)
        print('numFilesTarget', numFilesTarget)
        print('source ', sourceDir)

    if numFilesSource > numFilesTarget + 1:
        issueFlowCommandToPool(sourceDir, targetDir)

    elif numFilesSource == numFilesTarget + 1:
        fillFinalImageInDir(targetDir)


# def handleViews(aTaskdir, targetTask):
# views = os.listdir(aTaskdir)
#
# for aView in views:
#   viewPath = os.path.join(aTaskdir, aView)
#   targetView = os.path.join(targetTask, aView)
#
#   calculateOFBroxGPUAnSet(viewPath, targetView)


def handleTasksForPerson(aPersonDir, targetPerson):
    tasks = os.listdir(aPersonDir)

    for aTask in tasks:
        # handleViews(os.path.join(aPersonDir, aTask), \
        #   os.path.join(targetPerson, aTask))
        calculateOFBroxGPUAnSet(
            os.path.join(aPersonDir, aTask), os.path.join(targetPerson, aTask))


def handleSubjects(rootPath, targetRoot):
    persons = os.listdir(rootPath)
    # print('len(dirs)', len(persons))
    for aPerson in persons:
        # print('process person', aPerson)
        handleTasksForPerson(
            os.path.join(rootPath, aPerson), os.path.join(targetRoot, aPerson))


# root of the directory hierachy
sourcePath = '/home/afromero/datos/Databases/BP4D/Sequences'
targetBase = '/home/afromero/datos/Databases/BP4D/Sequences_Flow_'

handleSubjects(sourcePath, targetBase)
