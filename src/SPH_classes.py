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



# =============================================================================
#                                 Kernels

class Kernels():
    def __init__(self, h=0.1):
        # kernel factors
        self.h       = h
        self.poly6_fac      = 315.0/(np.pi * 64.0 * h**9)
        
        self.spiky_gradientFac      = -45.0/(np.pi        * h**6)
        
        self.viscosity_fac          =  15.0/(np.pi *  2.0 * h**3)
        self.viscosity_laplacianFac =  45.0/(np.pi        * h**6)
        
    # Poly 6
    # =====================================================
    def poly6_computeWeight(self,particleSystem, sqr_r):
        # i is the index of a given particle
        h = self.h
        fac = self.poly6_fac
        
        # Return weights
        return fac * (h**2 - sqr_r)**3
       
        
    # Spiky
    # =====================================================
    def spiky_computeGradientWeight(self, r, R):
        h = self.h
        fac = self.spiky_gradientFac
        
        return fac * (h - r)**2 * R/r
        
    
    # Viscosity
    # =====================================================
    def viscosity_computeLaplacianWeight(self, r):
        h = self.h
        fac = self.viscosity_laplacianFac
        
        return fac * (h - r)


    
#                                 Kernels
# =============================================================================




"""
                      ***********************************
"""




# =============================================================================
#                                 World
class World():
    def __init__(self,dt=0.001,
                 xmin = 0.0, xmax = 1.0,
                 ymin = 0.0, ymax = 1.0,
                 ):
        self.dt = dt
        self.g    = arr([0.0,-1.0])
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

#                                 World
# =============================================================================




"""
                      ***********************************
"""




# =============================================================================
#                             Particle System
class ParticleSystem():
    #                         Init
    # =========================================================
    def __init__(self,nx=10,ny=10,k=0.1,eta=1.0):
        # Constants
        # ================================
        # viscosity
        self.eta   = eta

        # Equation of state parameters
        self.k = k
        
        # Values for each particle
        # ================================
        # position
        self.nx = nx
        self.ny = ny
        n = nx*ny
        self.n  = n
        x,y = np.meshgrid(np.linspace(.25,.5,nx),np.linspace(.25,.5,ny))
        self.x = x.flatten()
        self.y = y.flatten()
        
        # velocity
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        
        # mass
        self.mass = np.ones(n)
        
        # density
        self.rho  = np.zeros(n)
        
        # Reference density
        self.rho0  = np.ones(n)
        
        # pressure
        self.P    = np.zeros(n)
        
        
        
        
        
        
    #                   Compute Distance
    # =========================================================
    def computeDistance(self,i,kernels):
        # define handy local variables
        x = self.x
        y = self.y
        h = kernels.h
        # compute distances
        sqr_r = (x-x[i])**2 + (y-y[i])**2
        J = sqr_r>h**2
        # /!\ Maybe I should ask a condition to exclude i from J
        
        sqr_r = sqr_r[J]
        r = np.sqrt(sqr_r)
        
        R = arr([ 
                 [x[i] - x[J] ],
                 [y[i] - y[J] ]
                ])

        return r, sqr_r, R, J
    
    
    
    
    
    
    
    #               Compute Density, Pressure
    # =========================================================
    def computeDensity(self,i,J,sqr_r,kernels):
        W = kernels.poly6_computeWeight(self,sqr_r)
        self.rho[i] = np.sum(self.mass[J]*W)
        
        
    def computePressure(self,i):
        self.P[i] = self.k * (self.rho[i]-self.rho0[i])
        
        
        
        
        
        
        
        
    #                     Compute Forces
    # =========================================================
    def computePressureForce(self,i,J,r,R,kernels):
        mj   = self.mass[J]
        Pi   = self.P[i]
        Pj   = self.P[J]
        rhoj = self.rho[J]
        
        gradW = kernels.spiky_computeGradientWeight(r,R)
        return np.sum( - (mj*( Pi+Pj/(2.0*rhoj) )*gradW) ) 
        
    
    def computeViscosityForce(self,i,J,r,R,kernels):
        mj   = self.mass[J]
        rhoj = self.rho[J]
        eta  = self.eta
        
        vxi  = self.vx[i]
        vxj  = self.vx[J]
        vyi  = self.vy[i]
        vyj  = self.vy[J]
        
        laplacianW = kernels.viscosity_computeLaplacianWeight(r)
        
        Visc_f_x = eta * np.sum( mj*(vxj-vxi)/rhoj*laplacianW )
        Visc_f_y = eta * np.sum( mj*(vyj-vyi)/rhoj*laplacianW )

        return arr([Visc_f_x , Visc_f_y])


    def computeGravityForce(self,i,world):
        return self.rho[i]*world.g
        

#                             Particle System
# =============================================================================

