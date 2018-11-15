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



# Create main objects
kernels = SPH.Kernels(h=0.2)
world   = SPH.World(xmax=10.0) 
ps = SPH.ParticleSystem(world,nx=11,ny=11)


nt = 50
plt.clf()


# =============================================================================
# Time loop
for it in range(nt):
    rr = np.zeros(ps.n)
    # Compute density and pressure
#     ========================================================
#    for iP in  range(1):
    for iP in range(ps.n):
        r, sqr_r, R, J = ps.computeDistance(iP,kernels)
        J = iP
        sqr_r = 0.0
        ps.computeDensity(iP,J,sqr_r,kernels)
        ps.computePressure(iP)
    
    
    # Compute forces and update position
    # ========================================================
    for iP in range(ps.n):
        r, sqr_r, R, J = ps.computeDistance(iP,kernels)
        
        # Compute forces
        # note: force components are returned in arrays(2)
        P_f     = ps.computePressureForce(iP,J,r,R,kernels)
        eta_f   = ps.computeViscosityForce(iP,J,r,R,kernels)
        g_f     = ps.computeGravityForce(iP,world)
        
        # Compute acceleration 
        # sum(forces) = m*a
        acc =1.0/ps.rho[iP] * ( P_f + eta_f + g_f )
        
        # Update velocity
        ps.vx[iP] += acc[0]*world.dt
        ps.vy[iP] += acc[1]*world.dt
        
        # Update velocity
        ps.x[iP] += ps.vx[iP]*world.dt
        ps.y[iP] += ps.vy[iP]*world.dt
    
    
    
#    # Render
#    # ========================================================
    plt.cla()
    plt.plot(ps.x,ps.y,'ob')
    plt.axis([world.xmin,world.xmax,world.ymin,world.ymax])
    plt.pause(0.00000000001)
#    plt.scatter(ps.x,ps.y,c=ps.P)
#    plt.colorbar()


# End Time loop
# =============================================================================
