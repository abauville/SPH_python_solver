#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:34:05 2018

@author: abauville

SPH solver serial version
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import array as arr
from time import time
from numba import jit
#import gr.pygr as gr


xmin = -.5
xmax = .5
ymin = -.5
ymax = .5

Wbox = xmax-xmin
Hbox = ymax-ymin

nx = 50
ny = 50
n  = nx*ny


dx = Wbox/(nx)
dy = Hbox/(ny)

h = 0.49*np.min([dx,dy])
h_sqr = h**2

#leftPad     =  .5*dx
#rightPad    = -.5*dx
#bottomPad   =  .5*dy
#topPad      = -.5*dy

leftPad     =  Wbox*.25
rightPad    = -Wbox*.25
bottomPad   =  .5*dy
topPad      = -Hbox*.25

x,y = np.meshgrid(
        np.linspace(xmin+leftPad,xmax+rightPad,nx),
        np.linspace(ymin+bottomPad,ymax+topPad,ny)
                 )
x = x.flatten()+(np.random.rand(n)-.5)*0.1*dx
y = y.flatten()+(np.random.rand(n)-.5)*0.1*dy


#vx = (np.random.rand(n)-.5) * dx/2.0
#vy = (np.random.rand(n)-.5) * dx/2.0

vx = np.zeros(n)
vy = np.zeros(n)

ax = np.zeros(n)
ay = np.zeros(n)

size = np.ones(n)*200.0
collide = np.zeros(n)
m = 1.0
k = 1.0
nt = 300
dt = 0.005

plt.figure(1)
plt.clf()
markers, = plt.plot(x,y,'.')
plt.xlim([xmin*1.05,xmax*1.05])
plt.ylim([ymin*1.05,ymax*1.05])


mass = np.ones(n)*dx*dy*np.pi / (np.max(x)-np.min(x)) / (np.max(y)-np.min(y))
rho  = np.zeros(n)
rho0 = np.ones(n)*1.0
P    = np.zeros(n)

# kernel radius
#h = 0.1


# Equation of state (EOS) factor
k = 1.0

# material properties
eta = 1.0

# coeff 2D
poly6_fac              =   4.0/(       np.pi * h**8)        
spiky_gradientFac      = -30.0/(       np.pi * h**5)
viscosity_laplacianFac =  20.0/( 3.0 * np.pi * h**5)

gravity = arr([0.0,-1.0])

boundPad = 0
boundDamping = 1.0


#@jit(nopython=True)
def computeDensityPressure(n,x,y,h_sqr,
                           poly6_fac,
                           rho,P,k,rho0):
    # Compute density and pressure
    # ========================================================
    for iP in range(n):
        # Compute distance sqr
        r_sqr = (x[iP]-x)**2 + (y[iP]-y)**2
        J = (r_sqr<h_sqr)        
        r_sqr = r_sqr[J]
            
        # Compute density    
        W = poly6_fac * (h_sqr - r_sqr)**3
        rho[iP] = mass[J] @ W
        
        # Compute Pressure
        P[iP] = k * (rho[iP]-rho0[iP])
            
#@jit(nopython=True)       
def updateAcceleration(n,x,y,
                       h,h_sqr,
                       spiky_gradientFac,viscosity_laplacianFac,
                       mass,P,rho,eta,gravity,
                       vx,vy,ax,ay    ):
    # Compute forces 
    # ========================================================
    for iP in range(n):
        r_sqr = (x[iP]-x)**2 + (y[iP]-y)**2
        J = (r_sqr<h_sqr)   
        J[iP] = False
        r_sqr = r_sqr[J]
        
        r = np.sqrt(r_sqr)
    
        R = arr([x[iP] - x[J] ,
                 y[iP] - y[J] ]) / r
        # Compute forces
        # note: force components are returned in arrays(2)
        
        # Compute pressure force
        gradW = spiky_gradientFac * (h - r)**2 * R
        Fac = - mass[J]*(P[iP]+P[J])/(2.0*rho[J])
        P_f = arr([ Fac @ gradW[0,:], Fac@gradW[1,:] ])
        
        
        # Compute viscous force
        laplacianW = viscosity_laplacianFac * (h - r)
        Fac = eta*mass[J]/rho[J]*laplacianW
        eta_f = arr([Fac @ (vx[J]-vx[iP]) , Fac @ (vy[J]-vy[iP])])

        # Compute gravity force
        g_f     = rho[iP]*gravity

        # Compute acceleration  (sum(forces) = m*a)
        ax[iP] =1.0/rho[iP] * ( P_f[0] + eta_f[0] + g_f[0] )
        ay[iP] =1.0/rho[iP] * ( P_f[1] + eta_f[1] + g_f[1] )
        
        
#@jit(nopython=True)          
def updatePosition(x,y,vx,vy,ax,ay,dt,
                   xmin,xmax,ymin,ymax):
    vx += ax*dt
    vy += ay*dt
    
    x += vx*dt
    y += vy*dt
    
    I = x<xmin+boundPad
    x[I]  = xmin+boundPad
    vx[I] = boundDamping*np.abs(vx[I])
    
    I = x>xmax-boundPad
    x[I]  = xmax-boundPad
    vx[I] = boundDamping*-np.abs(vx[I])

    I = y<ymin+boundPad
    y[I]  = ymin+boundPad
    vy[I] = boundDamping*np.abs(vy[I])
    
    I = y>ymax-boundPad
    y[I]  = ymax-boundPad
    vy[I] = boundDamping*-np.abs(vy[I])
    
    
#    for iP in range(n):
#        vx[iP] += ax[iP]*dt
#        vy[iP] += ay[iP]*dt
#        x[iP]  += vx[iP]*dt
#        y[iP]  += vy[iP]*dt
#
#        if x[iP]<xmin:
#            x[iP] = xmin
#            vx[iP] = -vx[iP]
#        if x[iP]>xmax:
#            x[iP] = xmax
#            vx[iP] = -vx[iP]
#        if y[iP]<ymin:
#            y[iP] = ymin
#            vy[iP] = -vy[iP]
#        if y[iP]>ymax:
#            y[iP] = ymax
#            vy[iP] = -vy[iP]
#            
            
            
simTime = 0
renderTime = 0

# pre compute
computeDensityPressure(n,x,y,h_sqr,
                           poly6_fac,
                           rho,P,k,rho0)
updateAcceleration(n,x,y,
                       h,h_sqr,
                       spiky_gradientFac,viscosity_laplacianFac,
                       mass,P,rho,eta,gravity,
                       vx,vy,ax,ay    )
updatePosition(x,y,vx,vy,ax,ay,dt,
               xmin,xmax,ymin,ymax)

for it in range(nt):
    tic = time()
    computeDensityPressure(n,x,y,h_sqr,
                           poly6_fac,
                           rho,P,k,rho0)
    updateAcceleration(n,x,y,
                       h,h_sqr,
                       spiky_gradientFac,viscosity_laplacianFac,
                       mass,P,rho,eta,gravity,
                       vx,vy,ax,ay    )
    updatePosition(x,y,vx,vy,ax,ay,dt,
                   xmin,xmax,ymin,ymax)
    simTime += time()-tic
    
    if it%10==0:
        tic = time()
        markers.set_data(x,y)
        plt.pause(1e-10)
        plt.title('timestep=%i/%i' % (it+1,nt))
#        print('it = %04d, fps=%.2f' % (it, 1.0/(time()-tic)))
        renderTime += time()-tic

print("sim time     = %.2f" % simTime)
print("render time  = %.2f" % renderTime)


