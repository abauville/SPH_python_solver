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
kernels = SPH.Kernels()
pSystem = SPH.ParticleSystem()
world   = SPH.World() 

nt = 1
plt.clf()


# =============================================================================
# Time loop
for it in range(nt):
    # Compute density and pressure
    # ========================================================
    for iP in range(pSystem.n):
        r, sqr_r, R, J = pSystem.computeDistance(iP,kernels)
        pSystem.computeDensity(iP,J,sqr_r,kernels)
        pSystem.computePressure(iP)
    
#    
#    
#    # Compute forces and update position
#    # ========================================================
#    for iP in range(pSystem.n):
#        r, sqr_r, R, J = pSystem.computeDistance(iP,kernels)
#        
#        # Compute forces
#        # note: force components are returned in arrays(2)
#        P_f     = pSystem.computePressureForce(iP,J,r,R,kernels)
#        eta_f   = pSystem.computeViscosityForce(iP,J,r,R,kernels)
#        g_f     = pSystem.computeGravityForce(iP,world)
#        
#        # Compute acceleration 
#        # sum(forces) = m*a
#        acc =1.0/pSystem.rho[iP] * ( P_f + eta_f + g_f )
#        
#        # Update velocity
#        pSystem.vx[iP] += acc[0]*world.dt
#        pSystem.vy[iP] += acc[1]*world.dt
#        
#        # Update velocity
#        pSystem.x[iP] += pSystem.vx[iP]*world.dt
#        pSystem.y[iP] += pSystem.vy[iP]*world.dt
#    
#    
#    
#    # Render
#    # ========================================================
#    plt.plot(pSystem.x,pSystem.y,'ob')
#    plt.scatter(pSystem.x,pSystem.y,pSystem.rho0)
    plt.plot(pSystem.rho)
    plt.axis([world.xmin,world.xmax,world.ymin,world.ymax])

# End Time loop
# =============================================================================


