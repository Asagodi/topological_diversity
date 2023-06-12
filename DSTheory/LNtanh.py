import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import numpy as np

import os, sys
currentdir = os.path.dirname(os.path.abspath(os.getcwd()))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir) 


def ReLU(x):
    return np.where(x<0,0,x)


import seaborn as sns

sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 1})

def ydot(y, alpha=2):
    return alpha*(1-y**2)*np.arctanh(y)
ys = np.arange(-1,1,.0001)
ydots = ydot(ys)
ydots[0]=0
ydots[-1]=0

fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)

plt.plot(ys, ydots)
axs.set(xlim=(-1.1, 1.1), ylim=(-1, 1))
axs.xaxis.set_major_locator(MaxNLocator(integer=True))
axs.yaxis.set_major_locator(MaxNLocator(integer=True))
axs.set_xlabel("$y$")
axs.set_ylabel("$\dot y$");

plt.savefig("figs/tanh_ydot.pdf", bbox_inches="tight");