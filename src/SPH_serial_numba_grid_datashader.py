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

import pandas as pd
import datashader as ds
#import gr
#import gr.pygr as gr








# Update Linked List
# ========================================================
@jit(nopython=True)
def updateLinkedList(x,y,
                     n,
                     xmin,ymin,
                     dxCell,dyCell,
                     nxCell,
                     linkHead,linkNext):

    linkHead[:] = -1
    for iP in range(n):
        Ix = np.floor((x[iP]-xmin)/dxCell) + 1
        Iy = np.floor((y[iP]-ymin)/dyCell) + 1

        I = np.int(Ix+Iy*nxCell)


        linkNext[iP] = linkHead[I]
        linkHead[I] = iP
#        linkTestNode[iP] = I


# Compute density and pressure
# ========================================================
@jit(nopython=True,parallel=True)
#@jit(nopython=True)
def computeDensityPressure(n,x,y,
                           xmin,ymin,
                           dxCell,dyCell,
                           nxCell,
                           linkHead,linkNext,
                           h_sqr,
                           poly6_fac,
                           mass,rho,P,k,rho0):


    for iP in prange(n):
        rho[iP] = 0.0
        xi = x[iP]
        yi = y[iP]
        # Get the index of the node closest to particle[iP]
        Ix = np.int(np.round((xi-xmin)/dxCell))
        Iy = np.int(np.round((yi-ymin)/dyCell))

        # Loop through neighboring cells
        for Ineigh in [Ix  + Iy  *nxCell,
                       Ix+1+ Iy  *nxCell,
                       Ix  +(Iy+1)*nxCell,
                       Ix+1+(Iy+1)*nxCell]:
            jP = linkHead[Ineigh]
            while (jP>-1): # Negative value = Null
                r_sqr = (xi-x[jP])**2 + (yi-y[jP])**2

                if r_sqr<h_sqr:
                    W = poly6_fac * (h_sqr - r_sqr)**3
                    rho[iP] += mass[jP] * W
                # end if dist
                jP = linkNext[jP]
            # end jP
        # end Ineigh
        P[iP] = k * (rho[iP]-rho0[iP])
    # end iP




# Compute forces
# ========================================================
@jit(nopython=True,parallel=True)
#@jit(nopython=True)
def updateAcceleration(n,x,y,
                       xmin,ymin,
                       nxCell,
                       dxCell,dyCell,
                       linkHead,linkNext,
                       h,h_sqr,
                       spiky_gradientFac,viscosity_laplacianFac,
                       mass,P,rho,eta,gravity,
                       vx,vy,ax,ay):


    for iP in prange(n):
        ax[iP] = 0.0;       ay[iP] = 0.0
        xi = x[iP];         yi = y[iP]
        axi = ax[iP];       ayi = ay[iP]
        vxi = vx[iP];       vyi = vy[iP]
        Pi = P[iP]

        # Get the index of the node closest to particle[iP]
        Ix = np.int(np.round((xi-xmin)/dxCell))
        Iy = np.int(np.round((yi-ymin)/dyCell))

        # Loop through neighboring cells
        for Ineigh in [Ix  + Iy  *nxCell,
                       Ix+1+ Iy  *nxCell,
                       Ix  +(Iy+1)*nxCell,
                       Ix+1+(Iy+1)*nxCell]:
            jP = linkHead[Ineigh]
            while (jP>=0): # Negative value = Null
                if jP==iP:
                    jP = linkNext[jP]
                    continue
                r_sqr = (xi-x[jP])**2 + (yi-y[jP])**2
                if r_sqr<h_sqr:
                    r = np.sqrt(r_sqr)
                    R = arr([(xi - x[jP])/r ,
                             (yi - y[jP])/r ])

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
                jP = linkNext[jP]
            # end jP
        # end Ineigh
        axi += rho[iP]*gravity[0]
        ayi += rho[iP]*gravity[1]

        ax[iP] = axi/rho[iP]
        ay[iP] = ayi/rho[iP]


    # end iP
##




# Update position
# ========================================================
@jit(nopython=True)
def updatePosition(x,y,vx,vy,ax,ay,dt,
                   xmin,xmax,ymin,ymax,
                   boundPad, boundDamping,n):

    for iP in prange(n):
        vx[iP] += ax[iP]*dt
        vy[iP] += ay[iP]*dt
        x[iP]  += vx[iP]*dt
        y[iP]  += vy[iP]*dt

        if x[iP]<xmin+boundPad:
            x[iP] = xmin+boundPad
            vx[iP] = boundDamping*np.abs(vx[iP])
        if x[iP]>xmax-boundPad:
            x[iP] = xmax-boundPad
            vx[iP] = boundDamping*-np.abs(vx[iP])
        if y[iP]<ymin+boundPad:
            y[iP] = ymin+boundPad
            vy[iP] = boundDamping*np.abs(vy[iP])
        if y[iP]>ymax-boundPad:
            y[iP] = ymax-boundPad
            vy[iP] = boundDamping*-np.abs(vy[iP])




# Main program
# ========================================================
#@profile
#def main():

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
nx = 30
ny = 45
n  = nx*ny



leftPad     =  Wbox*.25
rightPad    = -Wbox*.25
bottomPad   =  .2
topPad      = -Hbox*.25+.2
#
#    leftPad     =  .0
#    rightPad    = -Wbox*.75
#    bottomPad   =  .5*dy
#    topPad      = -Hbox*.25

#leftPad     =  dx*.5
#rightPad    = -dx*.5
#bottomPad   =  dy*.1
#topPad      = -dy*.5

x,y = np.meshgrid(
        np.linspace(xmin+leftPad,xmax+rightPad,nx),
        np.linspace(ymin+bottomPad,ymax+topPad,ny)
                 )

dx = x[0,1]-x[0,0]
dy = y[1,0]-y[0,1]

x = x.flatten()+(np.random.rand(n)-.5)*0.1*dx
y = y.flatten()+(np.random.rand(n)-.5)*0.1*dy



# Init particle velocity and acceleration
vx = np.zeros(n)
vy = np.zeros(n)

ax = np.zeros(n)
ay = np.zeros(n)



# Define material properties
k = 1.0
eta = .1
mass = np.ones(n)*dx*dy*np.pi / (np.max(x)-np.min(x)) / (np.max(y)-np.min(y))
rho  = np.zeros(n)
rho0 = np.ones(n)*1.0
P    = np.zeros(n)

# Define kernel radius and factors
h = .85*np.min([dx,dy])
h_sqr = h**2
# coeff 2D
poly6_fac              =   4.0/(       np.pi * h**8)
spiky_gradientFac      = -30.0/(       np.pi * h**5)
viscosity_laplacianFac =  20.0/( 3.0 * np.pi * h**5)


# Define time stuff
nt = 1
dt = 0.004


# Define linked list arrays
nxCell = np.int(np.floor(Wbox/dx))+2 # Include ghost cells (left and right)
nyCell = np.int(np.floor(Hbox/dy))+2
nCell = nxCell*nyCell
dxCell = Wbox/(nxCell-2)
dyCell = Hbox/(nyCell-2)
linkHead = np.zeros(nCell,dtype=int)
linkTestNode = np.zeros(n,dtype=int) # node to which the particles belongs
linkNext = np.zeros(n,dtype=int)





# Init figure
fig = plt.figure(2)
plt.clf()
gridX, gridY = np.meshgrid(np.linspace(xmin-dxCell,xmax+dxCell,nxCell+1),np.linspace(ymin-dyCell,ymax+dyCell,nyCell+1))



#markers, = plt.plot(x,y,'o',markersize=1.0,c=[.5,.7,.9])



#
#plt.xlim([xmin*1.05,xmax*1.05])
#plt.ylim([ymin*1.05,ymax*1.05])








simTime = 0
renderTime = 0

updateLinkedList(x,y,
                 n,
                 xmin,ymin,
                 dxCell,dyCell,
                 nxCell,
                 linkHead,linkNext)




#
#    plt.clf()
#    plt.plot(gridX,gridY,'k',linewidth=0.5)
#    plt.plot(gridX.T,gridY.T,'k',linewidth=0.5)
#    #        plt.fill([xmin,xmax,xmax,xmin],[ymin,ymin,ymax,ymax],color=[.9,.9,.9],linewidth=0.0)
#
#    plt.xlim([xmin-1.1*dxCell,xmax+1.1*dxCell])
#    plt.ylim([ymin-1.1*dyCell,ymax+1.1*dxCell])
#
#
#    plt.scatter(x,y,c=linkTestNode)
#    #        plt.scatter(x,y,c=P)
#    plt.set_cmap('tab20')
#    plt.colorbar()
#
#    plt.title('timestep=%i/%i' % (it+1,nt))
#    #        plt.pause(0.000000001)
#    plt.draw()
#    fig.canvas.flush_events()
#


# Call once for compilation (just useful for timing)
computeDensityPressure(n,x,y,
                           xmin,ymin,
                           dxCell,dyCell,
                           nxCell,
                           linkHead,linkNext,
                           h_sqr,
                           poly6_fac,
                           mass,rho,P,k,rho0)
updateAcceleration(n,x,y,
                   xmin,ymin,
                   nxCell,
                   dxCell,dyCell,
                   linkHead,linkNext,
                   h,h_sqr,
                   spiky_gradientFac,viscosity_laplacianFac,
                   mass,P,rho,eta,gravity,
                   vx,vy,ax,ay    )
updatePosition(x,y,vx,vy,ax,ay,dt,
               xmin,xmax,ymin,ymax,
               boundPad, boundDamping,n)



for it in range(nt):
    # Simulation
    # ============================
    tic = time.time()
#    print("Density")
    computeDensityPressure(n,x,y,
                           xmin,ymin,
                           dxCell,dyCell,
                           nxCell,
                           linkHead,linkNext,
                           h_sqr,
                           poly6_fac,
                           mass,rho,P,k,rho0)
#    print("Acceleration")
    updateAcceleration(n,x,y,
                   xmin,ymin,
                   nxCell,
                   dxCell,dyCell,
                   linkHead,linkNext,
                   h,h_sqr,
                   spiky_gradientFac,viscosity_laplacianFac,
                   mass,P,rho,eta,gravity,
                   vx,vy,ax,ay    )
#    print("Position")
    updatePosition(x,y,vx,vy,ax,ay,dt,
                   xmin,xmax,ymin,ymax,
                   boundPad, boundDamping,n)

#    print("LinkedList")
    updateLinkedList(x,y,
                     n,
                     xmin,ymin,
                     dxCell,dyCell,
                     nxCell,
                     linkHead,linkNext,
                     )
    simTime += time.time()-tic

    # Rendering
    # ============================
    if it==0:
        df = pd.DataFrame(np.array([x,y]).T,columns=('x','y'))
        refineFac = 4.0
        cvs = ds.Canvas(plot_width=int(nx*refineFac), plot_height=int(ny*refineFac),
                           x_range=(xmin,xmax), y_range=(ymin,ymax),
                           x_axis_type='linear', y_axis_type='linear')
        agg = cvs.points(df, 'x', 'y', ds.any())
        im = plt.imshow(agg.variable,extent=[xmin,xmax,ymin,ymax],origin='lower')

    if it%5==0:
        tic = time.time()
        df = pd.DataFrame(np.array([x,y]).T,columns=('x','y'))
        agg = cvs.points(df, 'x', 'y', ds.any())
        im.set_array(agg.variable)
#        markers.set_data(x,y)
#        plt.clf()
#        plt.plot(gridX,gridY,'k',linewidth=0.5)
#        plt.plot(gridX.T,gridY.T,'k',linewidth=0.5)
##        plt.fill([xmin,xmax,xmax,xmin],[ymin,ymin,ymax,ymax],color=[.9,.9,.9],linewidth=0.0)
#
#        plt.xlim([xmin-1.1*dxCell,xmax+1.1*dxCell])
#        plt.ylim([ymin-1.1*dyCell,ymax+1.1*dxCell])
#
#
##        plt.scatter(x,y,c=linkTestNode)
#        plt.scatter(x,y,c=P)
#        plt.set_cmap('inferno')
#        plt.colorbar()

        plt.title('timestep=%i/%i' % (it+1,nt))
#        plt.pause(0.000000001)
        plt.draw()
        fig.canvas.flush_events()
#        print('it = %04d, fps=%.2f' % (it, 1.0/(time.time()-tic)))
        renderTime += time.time()-tic


print("sim time     = %.2f" % simTime)
print("render time  = %.2f" % renderTime)


#main()
