# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:36:20 2018

@author: OKUDA akane
"""

import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import pandas as pd

fsize = 18 # Font size used in figures
tfsize = 24 # Font size of figure titles


def plamina_strat(max_x=5.0, min_x=0.0, dx=0.1,
                  max_t=10.0, min_t=0.0, dt=0.1,
                  DT = 1.5,
                  A=0.01,
                  D_init=1.0, D_last=0.01,
                  alpha_1=1.0, alpha_2=0.0,
                  loc = 1.2
                 ):
    """
    This is a function for calculating stratigraphic variation of
    grain size and erosional surfaces (laminae). Note that all parameters
    are dimensionless values using wave length and celerity of bedwaves
    
    input values are:
    max_x = 5 # maximum value of spatial coordinate of calculation domain
    min_x = 0 # minimum value of spatial coordinate of calculation domain
    dx = 0.1 #spatial interval
    max_t = 10 #maximum value of calculation time
    min_t = 0 #minimum value of calculation time
    dt = 0.1 #time interval
    DT = 1.5 #time inteval to output the bedform shape
    A = 0.01 # Amplitude of bedwaves
    D_init = 1.0 # Aggradation rate at t = 0
    D_last = 0.01 # Aggradation rate at t = max_t
    alpha_1 = 1.0 #coeffcient 1 for mean grain size
    alpha_2 = 0.0 #coeffcient 2 for mean grain size
    vmin_val = -2.0 #value for visualization of lamination
    vmax_val = 2.0 #value for visualization of lamination
    loc = 1.2 # Location of interest
    
    output values:
    t: time
    eta: bed elevation at a given location
    x: spatial coordinate
    T: time to output the bedform shape
    ETA: Bedform shape (T, x)
    gsize_df: Data frame of stratigraphy. Index labels are:
        "time": time
        "elevation": elevation
        "erosion": Did erosion occur at this horizon? (True or False)
        "gsize": grain size

    """
    
    # Calculate subordinate parameters
    loc_id = round(loc / dx) # Index of location of interest
    x = np.arange(min_x,max_x, dx) #spatial coordinate
    t = np.arange(min_t,max_t, dt) #time
    T = np.arange(min_t, max_t, DT) #time to output bedform shape 
    D = (D_last - D_init) / (max_t - min_t) * t + D_init #Aggradation rate at the time t
    DT = (D_last - D_init) / (max_t - min_t) * T + D_init #Aggradation rate at the time T
    
    # Create Numpy Arrays for storing data
    eta = np.zeros([len(t)]) #Bed elevation at a given location
    ETA = np.zeros([len(T),len(x)]) #Bed elevation for entire bedwaves
    gsize = np.zeros([len(t)]) #Mean grain size at a given location

    # Calculate Spatio-Temporal Evolution of Bed Elevation at a given location
    for j in range(len(t)):
        eta[j] = A * np.sin(2 * np.pi * (x[loc_id] - t[j])) + (D_init + D[j]) / 2 * t[j]
    
    # Calculate Spatio-Temporal Evolution of bed elevation in the all of calculation domain
    for i in range(len(T)):
        ETA[i,:] = A * np.sin(2 * np.pi * (x - T[i])) + (D_init + DT[i]) / 2 * T[i]
        
    # Calculate Spatio-Temporal Evolution of mean grain size at a given location
    # Add grain-size data at the highest elevation
    previous_elevation = eta[-1]
    gsize =  - alpha_1 * np.sin(2 * np.pi * (x[loc_id] - t[-1])) + alpha_2
    gsize_df = pd.DataFrame({"time":[t[-1]],
                             "elevation":[previous_elevation],
                             "erosion":[False],
                             "gsize":[gsize]}) # Data Frame for grain-size
    gsize_df = gsize_df.sort_values(by="elevation") #???
    # Add grain-size data from top to bottom for i in range(len(t) - 2):
    for i in range(len(t) - 2):
        current_elevation = eta[-(i+2)]
        # New data is added only when the elevation decreases
        if current_elevation < previous_elevation:
            gsize =  - alpha_1 * np.sin(2 * np.pi * (x[loc_id] - t[-(i+2)])) + alpha_2
            ndata = pd.Series([current_elevation, False, gsize, t[-(i+2)]],index=gsize_df.columns)
            gsize_df = gsize_df.append(ndata,ignore_index=True)
            previous_elevation = current_elevation
        else:
            gsize_df.iat[-1,1] = True
   
    
    # return calculated results
    return t, eta, x, T, ETA, gsize_df

def plot_bedwaves(x, T, ETA, savefig=True, fname="bedwave.pdf"):
    """
    function for ploting the entire shape of bedwaves
    
    input values:
    x: spatial coordinate
    T: time
    ETA: Bed elevation (T, x)
    savefig: flag for saving the figure in a file
    fname: file name to save the figure
    """
    plt.figure
    plt.title("Bedwaves", fontsize = tfsize)
    for k in range(len(T)):
        plt.plot(x,ETA[k,:],label='t = {:.1}'.format(T[k]))
        plt.xlabel("Dimensionless Distance", fontsize = fsize)
        plt.ylabel("Dimensionless Elevation", fontsize = fsize)
    plt.legend()
    if savefig:
        plt.savefig(fname)
    plt.show() 


def plot_elevation_change(t, eta, savefig=True, fname="bedelevation.pdf"):
    """
    a function for ploting the time development at the given location
    
    input values:
    t: time
    eta: Bed elevation at a given location
    savefig: flag for saving the figure in a file
    fname: file name to save the figure
    """
    plt.figure
    plt.title("Bed Elevation",fontsize=tfsize)
    plt.plot(t,eta)
    plt.xlabel("Dimensionless Time",fontsize=fsize)
    plt.ylabel("Dimensionless Elevation",fontsize=fsize)
    if savefig:
        plt.savefig(fname)
    plt.show()

def plot_strat(gsize_df, savefig=True, fname_lamina="lamina.pdf", fname_gsize="gsize.pdf", vmin_val=-2.0, vmax_val=2.0):
    """
    function for plotting the stratigraphy
    
    input values:
    gsize_df: data frame storing the stratigraphy
    """

    # plot the stratigraphy
    plt.figure
    plt.title("Grain Size Oscillation",fontsize=tfsize)
    plt.plot(gsize_df["gsize"],gsize_df["elevation"])
    plt.xlabel("Mean Grain Size ($\phi$)",fontsize=fsize)
    plt.ylabel("Dimensionless Elevation",fontsize=fsize)
    if savefig:
        plt.savefig(fname_gsize)
    plt.show()

    # Produce a map showing grain-size stratigraphy
    map_x = (np.max(gsize_df["elevation"])-np.min(gsize_df["elevation"]))/3.0
    map_y = np.linspace(np.min(gsize_df["elevation"]),np.max(gsize_df["elevation"]),100)
    gmap = np.zeros([len(map_y),1])
    f = ip.interp1d(gsize_df["elevation"],gsize_df["gsize"])
    gsize_ip = f(map_y)
    
    for i in range(len(gsize_ip)):
        gmap[-i,0] = gsize_ip[i]
    
    plt.figure
    plt.title("Lamination and Grain Size",fontsize=tfsize)
    plt.imshow(gmap,vmin=vmin_val,vmax=vmax_val,extent=(0,map_x,min(map_y),max(map_y)), cmap="jet_r")
    plt.colorbar()
    

    # plot erosional surface
    for i in range(gsize_df.shape[0]):
        if gsize_df.iat[i,1]:
            plt.plot([0,map_x],[gsize_df.iat[i,0],gsize_df.iat[i,0]],color='k')
    plt.ylabel("Dimensionless Elevation",fontsize=fsize)
    
    plt.tick_params(labelbottom=False)
    plt.xlim([0,map_x])
    plt.xticks([0,map_x])
    plt.ylim([gsize_df.iat[-1,0], gsize_df.iat[0,0]])
    
    if savefig:
        plt.savefig(fname_lamina)
    plt.show()