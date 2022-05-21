import os, matplotlib, sys
import numpy as np
import random
import torch
from termcolor import colored

#turn interactive mode off for plotting in background wihtout desktop
matplotlib.use('Agg')

print(colored("Bootstrapping DL Modules", 'blue'))
#find better way to do this..
thisDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(thisDir, '../thirdparty'))

print(colored("Included needed paths", 'blue'))
#print("filePatht: " + thisDir)
#for path in sys.path:
#    print(path)

def PlantSeed(manualSeed: int= 1337):
    manualSeed = manualSeed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(manualSeed)
    print(colored("Set Random-Seed to: " + str(manualSeed), 'blue'))

PlantSeed()

