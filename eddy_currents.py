'''I denne fil jeg forsøge, at beskrive samt at forstå eddy currents, hvor store er det som opstående af elektromagnetiske felter, og hvor meget EM udstråler de selv som vil modvirker den pågående magnetiske flux.
eddy currents er noget det opstår når et ledende materiale udsættes for et magnetisk felt. der opstår små strømme indeni materialet, som løber rundt om dermed udstråler dets eget magnetiske felt.'''

import pandas as pd
import numpy as np
import matplotlib as plt


B = 1e-5
mu0 = 4*np.pi*1e-7
sigma_gold = 4.11e7
sigma_silver = 6.30e7
sigma_copper = 5.96e7
sigma_alu = 3.5e7

material_radius = 0.1

material_size = np.array([1, 1, 1])*material_radius
frekvens = 100000  # Hz
omega = 2*np.pi*frekvens
A = np.pi*material_radius**2


'''
ved beregning af eddy currents bruges faradays lov om induktion
epsilon = - dø/dt
hvor epsilon = induced electromotive force (EMF) i volt
ø=magnetisk flux = B*A, hvor A er tværsnit af området ortogonalt til feltets retning i meter
dø/dt er ændringen af magnetisk flux over tid'''


'''så vi skal finde den mængde af EMF der bliver genereret i punktet, hvor det ledende materiale befinder sig og så kan vi estimere størrelsen af eddy currents i objektet derefter'''

epsilon_max = omega*B*A
print(epsilon_max)

length_current_path = 2*np.pi*material_radius
skin_depth_material = np.sqrt(2/(mu0*sigma_gold*omega))

R_gold = length_current_path / \
    (sigma_gold*length_current_path*skin_depth_material)

I_eddy = epsilon_max/R_gold

print(I_eddy)
print(R_gold)
