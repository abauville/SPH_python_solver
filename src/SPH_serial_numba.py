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
import time
from numba import jit, njit, prange

import gr
#import gr.pygr as gr

# Define space
xmin = -.5
xmax = .5
ymin = -.5
ymax = .5

Wbox = xmax-xmin
Hbox = ymax-ymin

boundPad = 0
boundDamping = 0.5

gravity = arr([0.0,-1.0])

# Init particle position
nx = 50
ny = 50
n  = nx*ny

dx = Wbox/(nx)
dy = Hbox/(ny)

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



# Init particle velocity and acceleration
vx = np.zeros(n)
vy = np.zeros(n)

ax = np.zeros(n)
ay = np.zeros(n)



# Define material properties
k = 1.0
eta = 1.0
mass = np.ones(n)*dx*dy*np.pi / (np.max(x)-np.min(x)) / (np.max(y)-np.min(y))
rho  = np.zeros(n)
rho0 = np.ones(n)*1.0
P    = np.zeros(n)

# Define kernel radius and factors
h = 0.5*np.min([dx,dy])
h_sqr = h**2
# coeff 2D
poly6_fac              =   4.0/(       np.pi * h**8)        
spiky_gradientFac      = -30.0/(       np.pi * h**5)
viscosity_laplacianFac =  20.0/( 3.0 * np.pi * h**5)


# Define time stuff
nt = 500
dt = 0.002


# Init figure
fig = plt.figure(1)
plt.clf()
markers, = plt.plot(x,y,'o',markersize=1.0)
plt.xlim([xmin*1.05,xmax*1.05])
plt.ylim([ymin*1.05,ymax*1.05])





# Compute density and pressure
# ========================================================
@jit(nopython=True,parallel=True)
def computeDensityPressure(n,x,y,h_sqr,
                           poly6_fac,
                           rho,P,k,rho0):
    
    for iP in prange(n):
        rho[iP] = 0.0
        for jP in range(n):
            # Compute distance sqr
            r_sqr = (x[iP]-x[jP])**2 + (y[iP]-y[jP])**2
            if r_sqr<h_sqr:                 
                # Compute density    
                W = poly6_fac * (h_sqr - r_sqr)**3
                rho[iP] += mass[jP] * W
                
            
        # Compute Pressure
        P[iP] = k * (rho[iP]-rho0[iP])





# Compute forces 
# ========================================================                        
@jit(nopython=True,parallel=True)   
def updateAcceleration(n,x,y,
                       h,h_sqr,
                       spiky_gradientFac,viscosity_laplacianFac,
                       mass,P,rho,eta,gravity,
                       vx,vy,ax,ay    ):
    
    for iP in prange(n):
        ax[iP] = 0.0;       ay[iP] = 0.0
        xi = x[iP];         yi = y[iP]
        axi = ax[iP];       ayi = ay[iP]
        vxi = vx[iP];       vyi = vy[iP]
        Pi = P[iP]
        for jP in range(n):
            if jP==iP:
                continue
            
            r_sqr = (xi-x[jP])**2 + (yi-y[jP])**2
            if r_sqr<h_sqr:                
                r = np.sqrt(r_sqr)
                R = arr([xi - x[jP] ,
                         yi - y[jP] ]) / r

                # Compute pressure force
                gradW = spiky_gradientFac * (h - r)**2 * R
                Fac = - mass[jP]*(Pi+P[jP])/(2.0*rho[jP])
                axi += Fac * gradW[0]
                ayi += Fac * gradW[1]
                
                # Compute viscous force
                laplacianW = viscosity_laplacianFac * (h - r)
                Fac = eta*mass[jP]/rho[jP]*laplacianW
                axi += Fac * (vx[jP]-vxi)
                ayi += Fac * (vy[jP]-vyi)
        
                
            # end if dist
        # end jP
        axi += rho[iP]*gravity[0]
        ayi += rho[iP]*gravity[1]
        
        ax[iP] = axi/rho[iP]
        ay[iP] = ayi/rho[iP]
        



# Update position
# ========================================================          
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
    
    
            
    
    
# Maint program
# ========================================================  
simTime = 0
renderTime = 0

# Call once for compilation (just useful for timing)
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
    # Simulation
    # ============================
    tic = time.time()
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
    simTime += time.time()-tic
    
    # Rendering
    # ============================
    if it%5==0:
        tic = time.time()
        
        markers.set_data(x,y)
        plt.title('timestep=%i/%i' % (it+1,nt))
        plt.draw()
        fig.canvas.flush_events()  
#        print('it = %04d, fps=%.2f' % (it, 1.0/(time.time()-tic)))
        renderTime += time.time()-tic


print("sim time     = %.2f" % simTime)
print("render time  = %.2f" % renderTime)


