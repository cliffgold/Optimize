# Importing necessary libraries
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import os

dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)

matrix = pd.read_csv('Mailbox/outbox/minimizer_Debug-8.csv')
xatol = pd.Series()
fatol = pd.Series()
gas_MW = pd.Series()
years  = matrix['Year'].max()
cases  = int(matrix['Year'].size / years)

for case in range(0, cases):
    xatol[case] = matrix.at[case*years, 'Xatol']
    fatol[case] = matrix.at[case*years, 'Fatol']
    matrix_part = matrix[int(case*years):int(case*years+8)]
    gas_MW[case] =  matrix_part['Gas_MW'].sum()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(xatol, fatol, gas_MW, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show() # or:
#fig.savefig('Mailbox/outbox/3D.png')

