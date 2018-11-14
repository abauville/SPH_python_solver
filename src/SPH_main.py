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
world   = SPH.World() 
ps = SPH.ParticleSystem(world,nx=11,ny=11)


nt = 1
plt.clf()


# =============================================================================
# Time loop
for it in range(nt):
    rr = np.zeros(ps.n)
    # Compute density and pressure
#     ========================================================
    for iP in  range(1):
#    for iP in range(ps.n):
        r, sqr_r, R, J = ps.computeDistance(iP,kernels)
        J = iP
        sqr_r = 0.0
        ps.computeDensity(iP,J,sqr_r,kernels)
        ps.computePressure(iP)
#        plt.plot(ps.rho)
#    
#    
#    # Compute forces and update position
#    # ========================================================
#    for iP in range(ps.n):
#        r, sqr_r, R, J = ps.computeDistance(iP,kernels)
#        
#        # Compute forces
#        # note: force components are returned in arrays(2)
#        P_f     = ps.computePressureForce(iP,J,r,R,kernels)
#        eta_f   = ps.computeViscosityForce(iP,J,r,R,kernels)
#        g_f     = ps.computeGravityForce(iP,world)
#        
#        # Compute acceleration 
#        # sum(forces) = m*a
#        acc =1.0/ps.rho[iP] * ( P_f + eta_f + g_f )
#        
#        # Update velocity
#        ps.vx[iP] += acc[0]*world.dt
#        ps.vy[iP] += acc[1]*world.dt
#        
#        # Update velocity
#        ps.x[iP] += ps.vx[iP]*world.dt
#        ps.y[iP] += ps.vy[iP]*world.dt
#    
#    
#    
#    # Render
#    # ========================================================
#    plt.plot(ps.x,ps.y,'ob')
    plt.scatter(ps.x,ps.y,c=ps.rho)
    plt.colorbar()
    plt.axis([world.xmin,world.xmax,world.ymin,world.ymax])

# End Time loop
# =============================================================================
#
#kernels = SPH.Kernels(h=np.sqrt(2.0))
#h = kernels.h
#n = 1000000
#r = np.linspace(0.0,kernels.h,n)
#dr = r[1]-r[0]
##r = h
#mass = np.ones(n)
#W = kernels.poly6_computeWeight(r**2)
#
#
#W2 = 315.0/(64.0 * np.pi * h**9) * (h**2-r**2)**3
#
#plt.plot(r/h,W,'b')
#plt.plot(r/h,W2,'r')
#rho = np.sum(mass*W)
#A = np.linspace(1.0,0.0,n)
#
#Aint = np.sum(mass/rho*A*W2)
#
#print(np.sum(mass*W2))
#print(np.sum(mass*W))
#print(np.sum(r)*dr)
#print(Aint)