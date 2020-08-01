#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from math import pi

np.random.seed(1)

def random_sphere(dim, N):
    x = np.random.normal(0, 1, size=(N,dim))
    r = np.power(np.random.random(size=N), 1./dim).reshape(N,1)
    shell=x/np.sqrt((x * x).sum(axis=1)).reshape(N,1)
    return r*shell

def calc_dists_from_org(sphere):
    return np.linalg.norm(sphere, axis=1)

def calc_dists_from_point(sphere,point):
    return np.linalg.norm(sphere-point, axis=1)

def sphere_volume(dim):
    return pi**(dim/2.)/(gamma(dim/2.+1))

def init(dim,ax):
    N=round(1000000*sphere_volume(dim))
    print(N)
    return random_sphere(dim, N)

def plt_dists_from_org(dim, ax):
    sphere = init(dim,ax)
    dists_from_org = calc_dists_from_org(sphere)
    ax.hist(
            dists_from_org, bins=50, range=(0,1), density=True,
            alpha=1.0, histtype='step', label='d='+str(dim))

def plt_dists_from_point(dim, ax, point):
    sphere = init(dim,ax)
    dists_from_point = calc_dists_from_point(sphere,point)
    ax.hist(
            dists_from_point, bins=50, range=(0,1), density=True,
            alpha=1.0, histtype='step', label='d='+str(dim))

fig = plt.figure()
ax = fig.add_subplot()

#plt_dists_from_org(2, ax)
#plt_dists_from_org(3, ax)
#plt_dists_from_org(5, ax)
#plt_dists_from_org(10, ax)
#plt_dists_from_org(15, ax)
#plt_dists_from_org(20, ax)

plt_dists_from_point(2, ax, [0.8,0])
plt_dists_from_point(3, ax, [0.8,0,0])
plt_dists_from_point(5, ax, [0.8,0,0,0,0])
plt_dists_from_point(10, ax,[0.8,0,0,0,0,0,0,0,0,0])
plt_dists_from_point(15, ax,[0.8,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
plt_dists_from_point(20, ax,[0.8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

ax.set_xlabel('r')
ax.set_ylabel('Freq')
ax.legend(loc='upper left')
plt.show()

