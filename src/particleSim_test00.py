#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:24:08 2018
@author: abauville

Basic particle simulator
"""
from numba import jit
import numpy as np
from time import time
renderer = 'mp'
if renderer=='mp':
    import matplotlib.pyplot  as plt
elif renderer=='gr':
    import gr.pygr.mlab  as plt
    import gr as gr
xmin = -.5
xmax = .5
ymin = -.5
ymax = .5

W = xmax-xmin
H = ymax-ymin

nx = 15
ny = 15
n  = nx*ny

dx = W/(nx)
dy = H/(ny)

h = 0.5*np.min([dx,dy])

x,y = np.meshgrid(
        np.linspace(xmin+.5*dx,xmax-.5*dx,nx),
        np.linspace(ymin+.5*dy,ymax-.5*dy,ny)
                 )
x = x.flatten()
y = y.flatten()

vx = (np.random.rand(n)-.5) * h
#vx = -x*.2
vy = (np.random.rand(n)-.5) * h

ax = np.zeros(n)
ay = np.zeros(n)

size = np.ones(n)*200.0
collide = np.zeros(n)
m = 1.0
k = 1.0
nt = 20
dt = 0.25

for it in range(nt):
    tic = time()
    color = np.zeros(n)
    @jit(nopython=True)
    for iP in range(n):
        ax[iP] = 0.0
        ay[iP] = 0.0
        for jP in range(n):
            if jP == iP:
                continue
            dist = np.sqrt((x[jP] - x[iP])**2 + (y[jP] - y[iP])**2)
#            print("ip=%i,jp=%i, dist=%.2f" % (iP,jP,dist))
            if dist<h:
                ax[iP] +=  - 1.0/m * k*(dist) * (x[jP]-x[iP])/dist
                ay[iP] +=  - 1.0/m * k*(dist) * (y[jP]-y[iP])/dist
                color[iP]=255.0
    for iP in range(n):
        vx[iP] += ax[iP]*dt
        vy[iP] += ay[iP]*dt
        x[iP] += vx[iP]*dt
        y[iP] += vy[iP]*dt

        if x[iP]<xmin:
            x[iP] = xmin
            vx[iP] = -vx[iP]
        if x[iP]>xmax:
            x[iP] = xmax
            vx[iP] = -vx[iP]
        if y[iP]<ymin:
            y[iP] = ymin
            vy[iP] = -vy[iP]
        if y[iP]>ymax:
            y[iP] = ymax
            vy[iP] = -vy[iP]
        
#    plt.hold(False)
    
#    if sum(color>0):
    
    # plt.scatter(x,y,np.ones(n)*200.0,color)
#    else:
#        plt.scatter(x,y,c=0)
    
    # gr.text(0.1,0.1,'fps=%.2f' % (time()-tic))
    plt.text(xmin+0.1,ymax-0.1,'fps=%.2f' % (time()-tic))
    plt.plot(x,y,'o')
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    
    if renderer=='mp':
        plt.pause(.00001)
        plt.cla()