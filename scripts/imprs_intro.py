# ================================================ #
# The Earth-System Box Model (ESBM):
# Reservoirs and Exchanges
#
# Author: Beatrice, Muxin
# Date: 07/07/2026
# ================================================ #


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs


# ============================================ #
# Box model for prognostic energy (i.e., Temp) 
# in surface ande deep ocean
# ============================================ #

# ============================= #
# Forward functions
# ============================= #
def euler_forward_func(dt, F_n, dF_n):
    """
    General Euler forward function,
    Will be used in the first time step
    """
    
    F_np1 = F_n + dt * dF_n
    
    return F_np1

def leapfrog_forward_func(dt, F_nm1, dF_n):
    """
    General Leapfrog forward function,
    Will be used from the 2nd timestep onward
    """
    
    F_np1 = F_nm1 + 2 * dt * dF_n
    
    return F_np1

# ============================= #
# Tendency functions
# ============================= #
# surface ocean
def T_s_tendency(T_s_n, T_D_n, C_a_n, dS0_n, C_a_0, alpha_0, S0_change=False):
    """
    the time derivative of surface ocean temperature 
    at current timestep

    Input:
    ------
    - T_s_n: surface ocean temperature at current timestep
    - T_D_n: deep ocean temperature at current timestep
    - C_a_n: atmospheric CO2 concentration at current timestep
    - C_a_0: pre-industrial atmospheric CO2 concentration
    - dS0_n: change in solar constant at current timestep
    - constant:
        - alpha_0: pre-industrial albedo
        - eta_H: heat exchange coefficient
        - S0: solar constant
        - lambda_: Radiativeresponse to surface temp anomaly 
            (climate sensitivity parameter)
        - lambda_star: Radiative response surface-deep ocean disequilibrium
        - beta: Radiative response to CO2 concentration
    """
    # define the constants:
    c_star = 10.8e9               # J K-1 m-2
    delta_surfocean = 0.015
    lambda_ = 1.77                # W m-1 K-2
    lambda_star = 0.75            # W m-1 K-2
    beta = 5.77                   # W m-2
    eta_H = 1                     # W m-2 K-1

    c_s = delta_surfocean * c_star

    radiative_damping = -lambda_ * T_s_n
    co2_forcing = beta * np.log(C_a_n / C_a_0)
    ocean_exchange = -eta_H * (T_s_n - T_D_n)
    disequilibrium_feedback = -lambda_star * (T_s_n - T_D_n)

    solar_forcing = 0
    if S0_change:
        solar_forcing = (1/4) * dS0_n * (1 - alpha_0)

    dT_s_n = (
        radiative_damping
        + co2_forcing
        + ocean_exchange
        + disequilibrium_feedback
        + solar_forcing
    ) / c_s

    return dT_s_n

# deep ocean
def T_D_tendency(T_s_n, T_D_n):
    """
    the time derivative of deep ocean temperature 
    at current timestep

    Input:
    ------
    - T_s_n: surface ocean temperature at current timestep
    - T_D_n: deep ocean temperature at current timestep
    - constant:
        - eta_H: heat exchange coefficient
    """
    # define the constants:
    eta_H = 1                     # W m-2 K-1
    c_star = 10.8e9               # J K-1 m-2
    delta_surfocean = 0.015

    # terms in tendency
    c_d = (1 - delta_surfocean) * c_star          
    dT_D_n = eta_H * ((T_s_n - T_D_n) / c_d)

    return dT_D_n


# ============================= #
# pre-requisites
# ============================= #
# time
seconds_per_year = 365 * 24 * 60 * 60
n_years = 100
dt = seconds_per_year          # timestep in seconds
nt = n_years + 1               # include year 0
time = np.arange(nt)  

# reference parameters
C_a_0 = 280.0                  # ppm, pre-industrial CO2
alpha_0 = 0.30                 # pre-industrial albedo

# prescribed forcings
dS0_values = np.zeros(nt)
C_a_values = 2 * C_a_0 * np.ones(nt)
# print(C_a_values)

# data array
ds = xr.Dataset(data_vars={
        "T_s": ("time", np.zeros(nt)),
        "T_D": ("time", np.zeros(nt)),
        "C_a": ("time", C_a_values),
        "dS0": ("time", dS0_values),},
    coords={"time": time},)

ds["time"].attrs["units"] = "years"
ds["T_s"].attrs["units"] = "K"
ds["T_D"].attrs["units"] = "K"
ds["C_a"].attrs["units"] = "ppm"
ds["dS0"].attrs["units"] = "W m-2"


# ============================= #
# time loop
# ============================= #
# initial condition
ds["T_s"][0] = 0.0
ds["T_D"][0] = 0.0

# loop over timestep: Leapfrog
for n in range(0, nt-1):
    # tendency at t=n (current)
    dT_s_n = T_s_tendency(
        T_s_n = ds["T_s"][n], 
        T_D_n = ds["T_D"][n], 
        C_a_n = ds["C_a"][n], 
        dS0_n = ds["dS0"][n], 
        C_a_0 = C_a_0, 
        alpha_0 = alpha_0, 
        S0_change=False)

    dT_D_n = T_D_tendency(
        T_s_n = ds["T_s"][n], 
        T_D_n = ds["T_D"][n])

    # update to t=n+1
    ds["T_s"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["T_s"][n], 
        dF_n = dT_s_n)

    ds["T_D"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["T_D"][n], 
        dF_n = dT_D_n)
    

# ============================= #
# time loop
# ============================= #
ds["T_s"].plot(label="surface ocean")
ds["T_D"].plot(label="deep ocean")
plt.xlabel("Time [years]")
plt.ylabel("Temperature anomaly [K]")
# plt.ylim([-10, 10])
plt.xlim([0,100])
plt.legend()
plt.show()