#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:24:08 2018
@author: abauville

SPH solver
"""



import matplotlib.pyplot as plt
import numpy as np
from numpy import array as arr
import SPH_classes as SPH
import time
from matplotlib.animation import FuncAnimation
#import gr.pygr as gr

# Create main objects
#kernels = SPH.Kernels(h=16.0)
#world   = SPH.World(xmax=800.0,ymax=600.0,dt=0.0008,boundPad=0.00,boundDamping=0.5, g = arr([.0,12000*-9.8]))
#ps = SPH.ParticleSystem(world,nx=11,ny=11,k=2000.0,rho0=1000.0,mass=65.0,eta=250.0)

kernels = SPH.Kernels(h=0.05)
world   = SPH.World(dt=0.005,boundPad=0.00,boundDamping=0.5)
ps = SPH.ParticleSystem(world,kernels,nx=15,ny=15,k=1.0,eta=2.0)

nt = 300
plt.clf()

line, = plt.plot(ps.x, ps.y,'.')
plt.axis([world.xmin-.05,world.xmax+.05,world.ymin-0.05,world.ymax+.05])
simTime = 0
renderTime = 0
# =============================================================================
# Time loop
for it in range(nt):
    tic = time.time()
    # Compute density and pressure
#     ========================================================
    for iP in range(ps.n):
        r, sqr_r, R, J = ps.computeDistance(iP,kernels)
        ps.computeDensity(iP,J,sqr_r,kernels)
        ps.computePressure(iP)
    
    
    # Compute forces and update position
    # ========================================================
    P_f_x_store = np.zeros(ps.n)
    P_f_y_store = np.zeros(ps.n)
    eta_f_x_store = np.zeros(ps.n)
    eta_f_y_store = np.zeros(ps.n)
#    P_f_store = np.zeros(n)
    for iP in range(ps.n):
#        print(iP)
        r, sqr_r, R, J = ps.computeDistance(iP,kernels,excludeSelf=True)
        
        # Compute forces
        # note: force components are returned in arrays(2)
        P_f     = arr([.0,.0])
        eta_f   = arr([.0,.0])
        g_f     = arr([.0,.0])
        
        P_f     = ps.computePressureForce(iP,J,r,R,kernels)
        eta_f   = ps.computeViscosityForce(iP,J,r,R,kernels)
        g_f     = ps.computeGravityForce(iP,world)

        
        P_f_x_store[iP] = P_f[0]
        P_f_y_store[iP] = P_f[1]
#        eta_f_x_store[iP] = eta_f[0]
#        eta_f_y_store[iP] = eta_f[1]
        
        # Compute acceleration 
        # sum(forces) = m*a
        acc =1.0/ps.rho[iP] * ( P_f + eta_f + g_f )
        
        # Update velocity
        ps.vx[iP] += acc[0]*world.dt
        ps.vy[iP] += acc[1]*world.dt
        
        
    
        
    # Update position and enforce BC
    # ========================================================
    # Update position
    ps.x += ps.vx*world.dt
    ps.y += ps.vy*world.dt
    
    I = ps.x<world.xmin+world.boundPad
    ps.x[I]  = world.xmin+world.boundPad
    ps.vx[I] = world.boundDamping*np.abs(ps.vx[I])
#    ps.vx[I] *= world.boundDamping
    
    I = ps.x>world.xmax-world.boundPad
    ps.x[I]  = world.xmax-world.boundPad
    ps.vx[I] = world.boundDamping*-np.abs(ps.vx[I])
#    ps.vx[I] *= world.boundDamping
    
#    
    I = ps.y<world.ymin+world.boundPad
#    if np.sum(I>0):
#        stop = 0
    ps.y[I]  = world.ymin+world.boundPad
    ps.vy[I] = world.boundDamping*np.abs(ps.vy[I])
#    ps.vy[I] *= world.boundDamping
    
    I = ps.y>world.ymax-world.boundPad
    ps.y[I]  = world.ymax-world.boundPad
    ps.vy[I] = world.boundDamping*-np.abs(ps.vy[I])
#    ps.vy[I] *= world.boundDamping
    
#    # Render
#    # ========================================================
    toc = time.time()
    simTime += toc-tic
    
    if it%5==0:
        tic = time.time()
#    plt.cla()
        line.set_xdata(ps.x)
        line.set_ydata(ps.y)
#        plt.clf()
        
#        phi = np.linspace(0,2.0*np.pi,32)
#        plt.plot(ps.x[0]+kernels.h*np.cos(phi),ps.y[0]+kernels.h*np.sin(phi),color=[.8,.8,.9])
#        plt.scatter(ps.x,ps.y,c=ps.rho-ps.rho0)
#        plt.scatter(ps.x,ps.y,c=P_f_x_store)
#        plt.colorbar()
        plt.title('timestep=%i/%i' % (it+1,nt))
#        plt.draw()
#        plt.axis([world.xmin-.05,world.xmax+.05,world.ymin-0.05,world.ymax+.05])
        plt.pause(0.00000000001)
        toc = time.time()
        renderTime += toc-tic

# End Time loop
# =============================================================================

print("sim time     = %.2f" % simTime)
print("render time  = %.2f" % renderTime)