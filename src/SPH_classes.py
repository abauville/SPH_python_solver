#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:24:08 2018
@author: abauville

Set of classes for the SPH solver
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
        
        self.h_sqr = h**2
        
#        # coeff 3D
#        self.poly6_fac              = 315.0/(64.0 * np.pi * h**9)     
#        self.spiky_gradientFac      = -45.0/(       np.pi * h**6)
#        self.viscosity_laplacianFac =  45.0/(       np.pi * h**6)
        
        # coeff 2D
        self.poly6_fac              =   4.0/(       np.pi * h**8)        
        self.spiky_gradientFac      = -30.0/(       np.pi * h**5)
        self.viscosity_laplacianFac =  20.0/( 3.0 * np.pi * h**5)
        
    # Poly 6
    # =====================================================
    def poly6_computeWeight(self,r_sqr):        
        return self.poly6_fac * (self.h_sqr - r_sqr)**3
       
        
    # Spiky
    # =====================================================
    def spiky_computeGradientWeight(self, r, R):
        return self.spiky_gradientFac * (self.h - r)**2 * R
        
    
    # Viscosity
    # =====================================================
    def viscosity_computeLaplacianWeight(self, r):
        return self.viscosity_laplacianFac * (self.h - r)


    
#                                 Kernels
# =============================================================================




"""
                      ***********************************
"""




# =============================================================================
#                                 World
class World():
    def __init__(self,dt=0.01,
                 xmin = 0.0, xmax = 1.0,
                 ymin = 0.0, ymax = 1.0,
                 boundPad = 0.0, boundDamping = -.5,
                 g = arr([0.0,-1.0])):
        self.dt = dt
        self.g    = g
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        self.boundDamping = boundDamping
        self.boundPad = boundPad

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
    def __init__(self,world,kernels,nx=10,ny=10,k=0.1,eta=1.0,mass=1.0,rho0=1.0):
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
##        x,y = np.meshgrid(np.linspace(.25,.5,nx),np.linspace(.25,.5,ny))
##        dx = (world.xmax-world.xmin)/(nx+1)
##        dy = (world.ymax-world.ymin)/(ny+1)
#        dx = kernels.h
#        dy = kernels.h
##        x,y = np.meshgrid(np.linspace(world.xmin+dx,world.xmax-dx,nx),np.linspace(world.ymin+dy,world.ymax-dy,ny))
#        x0 = world.xmin+(world.xmax-world.xmin)/4.0
#        y0 = world.ymin+dy
#        x,y = np.meshgrid(np.linspace(x0,x0+nx*dx,nx),np.linspace(y0,y0+ny*dy,ny))
##        x,y = np.meshgrid(np.linspace(world.xmin+world.xmax/4.0,world.xmax/2.0,nx),np.linspace(world.ymin+16.0,world.ymin+16.0+16.0*ny,ny))
#        self.x = x.flatten() + np.random.rand(n)*dx*0.25
#        self.y = y.flatten() + np.random.rand(n)*dy*0.25
#        
        
        Wbox = world.xmax - world.xmin
        Hbox = world.ymax - world.ymin
        dx = Wbox/(nx)
        dy = Hbox/(ny)
        
        print("h = %.5g" %( 0.49*np.min([dx,dy])))

        leftPad     =  Wbox*.25
        rightPad    = -Wbox*.25
        bottomPad   =  .5*dy
        topPad      = -Hbox*.25
        
        
        x,y = np.meshgrid(
                np.linspace(world.xmin+leftPad,world.xmax+rightPad,nx),
                np.linspace(world.ymin+bottomPad,world.ymax+topPad,ny)
                         )
        self.x = x.flatten()+(np.random.rand(n)-.5)*0.1*dx
        self.y = y.flatten()+(np.random.rand(n)-.5)*0.1*dy
        
        # velocity
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        
        # mass
        self.mass = np.ones(n)*dx*dy*np.pi / (np.max(x)-np.min(x)) / (np.max(y)-np.min(y))
#        self.mass = np.ones(n)*mass
        # density
        self.rho  = np.zeros(n)
        
        # Reference density
        self.rho0  = np.ones(n)*rho0
        
        # pressure
        self.P    = np.zeros(n)
        
        
        
        
        
        
    #                   Compute Distance
    # =========================================================
    #@profile
    def computeDistanceSqr(self,i,kernels):
        # define handy local variables
        x = self.x
        y = self.y
        h = kernels.h
        # compute distances
        r_sqr = (x[i]-x)**2 + (y[i]-y)**2
        J = (r_sqr<h**2)
        
        r_sqr = r_sqr[J]
        return r_sqr, J
    
#    @profile
    def computeDistance(self,i,kernels):
        # define handy local variables
        x = self.x
        y = self.y
        xi = x[i]
        yi = y[i]
        # compute distances
        r_sqr = (xi-x)**2 + (yi-y)**2
        J = (r_sqr<kernels.h_sqr)
        J[i] = False
        
        r_sqr = r_sqr[J]
        r = np.sqrt(r_sqr)
        
        R = arr([ 
                 xi - x[J] ,
                 yi - y[J] 
                ]) / r
    

        return r, r_sqr, R, J
    
    
    
    
    #               Compute Density, Pressure
    # =========================================================
    #@profile
    def computeDensity(self,i,J,r_sqr,kernels):
        W = kernels.poly6_computeWeight(r_sqr)
        self.rho[i] = self.mass[J] @ W
        
        
    def computePressure(self,i):
        self.P[i] = self.k * (self.rho[i]-self.rho0[i])
        
        
        
        
        
        
        
        
    #                     Compute Forces
    # =========================================================
    #@profile
    def computePressureForce(self,i,J,r,R,kernels):
        gradW = kernels.spiky_computeGradientWeight(r,R)
        Fac = - self.mass[J]*(self.P[i]+self.P[J])/(2.0*self.rho[J])
#        return np.sum(- (self.mass[J]*(self.P[i]+self.P[J])/(2.0*self.rho[J])*gradW) , 1) 
        return arr([ Fac @ gradW[0,:], Fac@gradW[1,:] ])

        
    #@profile
    def computeViscosityForce(self,i,J,r,kernels):        
        laplacianW = kernels.viscosity_computeLaplacianWeight(r)        
        Fac = self.eta*self.mass[J]/self.rho[J]*laplacianW
        return arr([Fac @ (self.vx[J]-self.vx[i]) , Fac @ (self.vy[J]-self.vy[i])])


    def computeGravityForce(self,i,world):
        return self.rho[i]*world.g
        

#                             Particle System
# =============================================================================

