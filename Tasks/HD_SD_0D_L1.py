import torch

import sys

from task import *
from build import *
from test_funcs import *

# Task Dependencies
from Tasks.HD_SD_0D import *

HD_SD_0D_L1_TASK = Task.named('HD_SD-0D', 
                         name='HD_SD-0D-L1',
                         loss_func=rate_l2_weight_l1_loss_func,
                         register=True)



