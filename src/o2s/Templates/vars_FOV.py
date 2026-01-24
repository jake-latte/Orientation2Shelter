import torch
import numpy as np
from typing import Dict, Tuple

import o2s
import o2s.Templates.vars_0D as template_0D

default_params = {
    'fov': np.pi/4
}


def add_fov_var(config, vars):
    fov = config.fov

    sd = vars['sd']

    shelter_in_fov = torch.where((sd <= fov/2) | (sd >= 2*np.pi-fov/2), torch.ones_like(sd), torch.zeros_like(sd)).to(torch.bool)

    vars['shelter_in_FOV'] = shelter_in_fov
    return vars

