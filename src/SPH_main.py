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


#@profile
def main():
    # Create main objects
    kernels = SPH.Kernels(h=0.032)
    world   = SPH.World(dt=0.005,boundPad=0.00,boundDamping=1.0)
    ps = SPH.ParticleSystem(world,kernels,nx=15,ny=15,k=10.0,eta=1.0)
    
    nt = 300
    plt.figure(2)
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
        # ========================================================
        for iP in range(ps.n):
            r_sqr, J = ps.computeDistanceSqr(iP,kernels)
            ps.computeDensity(iP,J,r_sqr,kernels)
            ps.computePressure(iP)
        
        
        # Compute forces and update position
        # ========================================================
        for iP in range(ps.n):
            r, r_sqr, R, J = ps.computeDistance(iP,kernels)
            
            # Compute forces
            # note: force components are returned in arrays(2)
            P_f     = ps.computePressureForce(iP,J,r,R,kernels)
            eta_f   = ps.computeViscosityForce(iP,J,r,kernels)
            g_f     = ps.computeGravityForce(iP,world)
    
            
            # Compute acceleration  (sum(forces) = m*a)
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
        
        I = ps.x>world.xmax-world.boundPad
        ps.x[I]  = world.xmax-world.boundPad
        ps.vx[I] = world.boundDamping*-np.abs(ps.vx[I])

        I = ps.y<world.ymin+world.boundPad
        ps.y[I]  = world.ymin+world.boundPad
        ps.vy[I] = world.boundDamping*np.abs(ps.vy[I])
        
        I = ps.y>world.ymax-world.boundPad
        ps.y[I]  = world.ymax-world.boundPad
        ps.vy[I] = world.boundDamping*-np.abs(ps.vy[I])
        
        toc = time.time()
        simTime += toc-tic
        
        # Render
        # ========================================================
    
        
        if it%10==0:
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
    
main()