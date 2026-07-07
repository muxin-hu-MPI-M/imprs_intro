# ================================================ #
# The Earth-System Box Model (ESBM):
# Reservoirs and Exchanges
#
# Author: Beatrice, Muxin
# Date: 07/07/2026
# ================================================ #

# %% 
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

# %% 
# Forward functions
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

# %% 
# Temperature tendency functions
# T for surface ocean (unit: K yr-1)
def T_s_tendency(T_s_n, T_D_n, C_a_n, dS0_n, Constants, S0_change=False):
    """
    the time derivative of surface ocean temperature 
    at current timestep

    Input:
    ------
    - T_s_n: surface ocean temperature at current timestep
    - T_D_n: deep ocean temperature at current timestep
    - C_a_n: anomalous atmospheric CO2 concentration 
             at current timestep (compare to pre-industrial)
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
    c_star = Constants["c_star"]                   # J K-1 m-2
    delta_surfocean = Constants["delta_surfocean"]
    lambda_ = Constants["lambda_"]                 # W m-2 K-1
    lambda_star = Constants["lambda_star"]         # W m-2 K-1
    beta = Constants["beta"]                       # W m-2
    eta_H = Constants["eta_H"]                     # W m-2 K-1
    CC_a_0 = Constants["CC_a_0"]                   # GtC, pre-industrial atmospheric carbon

    c_s = delta_surfocean * c_star

    radiative_damping = -lambda_ * T_s_n
    co2_forcing = beta * np.log(C_a_n / CC_a_0 + 1)
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
    ) / c_s  # K s-1

    return dT_s_n * (360 * 24 * 60 * 60)  

# T for deep ocean (unit: K yr-1)
def T_D_tendency(T_s_n, T_D_n, Constants):
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
    eta_H = Constants["eta_H"]                      # W m-2 K-1
    c_star = Constants["c_star"]                    # J K-1 m-2
    delta_surfocean = Constants["delta_surfocean"] 

    # terms in tendency
    c_d = (1 - delta_surfocean) * c_star          
    dT_D_n = eta_H * ((T_s_n - T_D_n) / c_d) # K s-1

    return dT_D_n * (360 * 24 * 60 * 60)  

# %%
# Carbon tendency equations
# land carbon tendency (units: GtC yr-1)
def C_l_tendency(C_a_n, C_l_n, T_s_n, Constants):
    """
    the time derivative of land carbon 
    at current timestep

    Input:
    ------
    - C_a_n: atmospheric carbon anonaly (GtC) at current timestep
    - C_l_n: land carbon anomaly (GtC) at current timestep
    - T_s_n: surface ocean temperature at current timestep
    - constant:
        - PI_0 : pre-industrial net primary production (NPP)
        - beta_PI: CO2 fertilization effect on NPP
        - CC_a_0: pre-industrial atmospheric carbon (GtC)
        - tau_l_0: pre-industrial land carbon turnover time (lifetime) (yr)
        - Chi: Q_10 factor for temperature sensitivity of land carbon turnover
        - CC_l_0: pre-industrial land carbon (GtC)
    """
    # constants
    PI_0 = Constants["PI_0"]                    # GtC yr-1
    beta_PI = Constants["beta_PI"]              # dimensionless
    CC_a_0 = Constants["CC_a_0"]                # GtC, pre-industrial atmospheric carbon
    tau_l_0 = Constants["tau_l_0"]              # yr
    Chi = Constants["Chi"]                      # dimensionless
    CC_l_0 = Constants["CC_l_0"]                # GtC

    # prognostic total land carbon (GtC)
    CC_l_n = CC_l_0 + C_l_n

    # terms
    NPP = PI_0 * (1 + beta_PI * np.log(C_a_n / CC_a_0 + 1))
    Respiration = - (CC_l_n / tau_l_0) * Chi ** (T_s_n / 10)

    # tendency
    dC_l_n = (
        NPP
        + Respiration
    ) # GtC yr-1

    return dC_l_n

# surface ocean carbon tendency (units: GtC yr-1)
def C_s_tendency(C_a_n, C_s_n, C_d_n, Constants):
    """
    the time derivative of surface ocean carbon 
    at current timestep

    Input:
    ------
    - C_a_n: atmospheric carbon anonaly (GtC) at current timestep
    - C_s_n: surface ocean carbon anonaly (GtC) at current timestep
    - C_d_n: deep ocean carbon anonaly (GtC) at current timestep
    - constant:
        - gamma : Air sea gas exchange coefficient
        - kappa_a: Atmospheric carbon intensity factor (GtC ppm-1)
        - kappa_o: Ocean carbon intensity factor (GtC ppm-1)
        - eta_c: surface-deep ocean carbon exchange coefficient
        - delta_surfocean: fraction of surface ocean in total ocean
    """
    # constants
    gamma = Constants["gamma"]                                        # GtC yr-1 ppm-1
    kappa_a = Constants["kappa_a"]                                    # GtC ppm-1
    eta_c = Constants["eta_c"]                                        # yr-1
    delta_surfocean = Constants["delta_surfocean"]                    # fraction
    kappa_o = Constants["kappa_o"]                                    # GtC ppm-1

    # terms
    uptake_of_atmos_C_by_surf_ocean = gamma * (C_a_n / kappa_a - C_s_n / kappa_o)
    update_of_surf_ocean_C_by_deep_ocean = - eta_c * (C_s_n/delta_surfocean - C_d_n/(1-delta_surfocean))

    # tendency
    dC_s_n = (
        uptake_of_atmos_C_by_surf_ocean
        + update_of_surf_ocean_C_by_deep_ocean
    ) # GtC yr-1

    return dC_s_n

# deep ocean carbon tendency (units: GtC yr-1)
def C_d_tendency(C_s_n, C_d_n, Constants):
    """
    the time derivative of surface ocean carbon 
    at current timestep

    Input:
    ------
    - C_s_n: surface ocean carbon anonaly (GtC) at current timestep
    - C_d_n: deep ocean carbon anonaly (GtC) at current timestep
    - constant:
        - eta_c: surface-deep ocean carbon exchange coefficient
        - delta_surfocean: fraction of surface ocean in total ocean
    """
    # constants
    eta_c = Constants["eta_c"]                      # yr-1
    delta_surfocean = Constants["delta_surfocean"]  # fraction

    # terms
    update_of_surf_ocean_C_by_deep_ocean = eta_c * (C_s_n/delta_surfocean - C_d_n/(1-delta_surfocean))

    # tendency
    dC_d_n = (
        update_of_surf_ocean_C_by_deep_ocean
    ) # GtC yr-1

    return dC_d_n

# atmospheric carbon tendency (units: GtC yr-1)
def C_a_tendency(n, dC_l_n, dC_s_n, dC_d_n, Constants, if_emission=True):
    """
    the time derivative of atmospheric carbon 
    at current timestep

    Input:
    ------
    - I: anthropogenic carbon emission (GtC yr-1) at current timestep
    - dC_l_n: land carbon tendency (GtC yr-1) at current timestep
    - dC_s_n: surface ocean carbon tendency (GtC yr-1) at current timestep
    - dC_d_n: deep ocean carbon tendency (GtC yr-1) at current timestep
    """
    # terms
    uptake_of_atmos_C_by_land = dC_l_n
    uptake_of_atmos_C_by_surf_ocean = dC_s_n
    uptake_of_surf_ocean_C_by_deep_ocean = dC_d_n

    # anthropogenic carbon emission (Eqs.7.17)
    A_tot = Constants["A_tot"]        # GtC
    t_opt = Constants["t_opt"]         # yr
    I = 0
    if if_emission:
        I = (A_tot / 50) * (2.5 * np.exp((t_opt - time[n]) / 50)) / (1 + 2.5 * np.exp((t_opt - time[n]) / 50))**2
    
    # tendency
    dC_a_n = (
        I
        - uptake_of_atmos_C_by_land
        - uptake_of_atmos_C_by_surf_ocean
        - uptake_of_surf_ocean_C_by_deep_ocean
    ) # GtC yr-1

    return dC_a_n

# %% 
# Pre-requisites
# time
n_years = 2000     
dt = 1                          # timestep: 1 year
nt = n_years + 1                # include year 0
time = np.arange(nt)  

# Constants
alpha_0 = 0.40                  # pre-industrial albedo
c_star = 10.8e9                 # J K-1 m-2
delta_surfocean = 0.015
lambda_ = 1.77                  # W m-2 K-1
lambda_star = 0.75              # W m-2 K-1
beta = 5.77                     # W m-2
eta_H = 1e-1                    # W m-2 K-1
PI_0 = 60.0                     # GtC yr-1
beta_PI = 0.4                   # dimensionless
tau_l_0 = 41                    # yr
Chi = 1.8                       # dimensionless
gamma = 0.02                                # GtC yr-1 ppm-1
kappa_a = 2.12                              # GtC ppm-1
eta_c = (60e-12) * (365 * 24 * 60 * 60)     # yr-1
CC_a_0 = 280.0 * kappa_a                    # GtC, pre-industrial atmospheric carbon
CC_l_0 = PI_0 * tau_l_0                     # GtC, pre-industrial land carbon
kappa_o = kappa_a * ((1-(1/7))/(1/7)) * delta_surfocean   # GtC ppm-1
A_tot = 5000                                              # GtC
t_opt = 250                                               # yr

Constants = {
    "alpha_0": alpha_0,
    "c_star": c_star,
    "delta_surfocean": delta_surfocean,
    "lambda_": lambda_,
    "lambda_star": lambda_star,
    "beta": beta,
    "eta_H": eta_H,
    "PI_0": PI_0,
    "beta_PI": beta_PI,
    "CC_a_0": CC_a_0,
    "CC_l_0": CC_l_0,
    "tau_l_0": tau_l_0,
    "Chi": Chi,
    "gamma": gamma,
    "kappa_a": kappa_a,
    "eta_c": eta_c,
    "kappa_o": kappa_o,
    "A_tot": A_tot,
    "t_opt": t_opt,}

# prescribed forcings
# solar constant
dS0_values = np.zeros(nt)

# ONLY FOR TEMPERATURE MODEL
# carbon concentration [ppm] -> ONLY FOR TEMPAURE MODEL
# C_a_values = np.zeros(nt)
# C_a_values = 2 * CC_a_0 * np.ones(nt) - CC_a_0 # CO2 concentration ANOMALY
# FOR FULL MODEL, carbon forcing is given in I in C_a_tendency function

# data array
ds = xr.Dataset(data_vars={
        "T_s": ("time", np.zeros(nt)),
        "T_D": ("time", np.zeros(nt)),
        "C_a": ("time", np.zeros(nt)), # will be updated by carbon model
        "dS0": ("time", dS0_values),
        "C_l": ("time", np.zeros(nt)),
        "C_s": ("time", np.zeros(nt)),
        "C_d": ("time", np.zeros(nt)),},
    coords={"time": time},)

ds["time"].attrs["units"] = "years"
ds["T_s"].attrs["units"] = "K"
ds["T_D"].attrs["units"] = "K"
ds["C_a"].attrs["units"] = "GtC"
ds["dS0"].attrs["units"] = "W m-2"
ds["C_l"].attrs["units"] = "GtC"
ds["C_s"].attrs["units"] = "GtC" 
ds["C_d"].attrs["units"] = "GtC" 

# %% 
# time loop
# initial condition for ANOMALY
ds["T_s"][0] = 0.0
ds["T_D"][0] = 0.0
ds["C_a"][0] = 0.0
ds["C_l"][0] = 0.0
ds["C_s"][0] = 0.0
ds["C_d"][0] = 0.0

for n in range(0, nt-1):
    # tendency at t=n (current)
    # T_s_n, T_D_n, C_a_n, dS0_n, Constants, S0_change=False
    dT_s_n = T_s_tendency(
        T_s_n = ds["T_s"][n], 
        T_D_n = ds["T_D"][n], 
        C_a_n = ds["C_a"][n], 
        dS0_n = ds["dS0"][n], 
        Constants = Constants, 
        S0_change=False)
    # T_s_n, T_D_n, Constants
    dT_D_n = T_D_tendency(
        T_s_n = ds["T_s"][n], 
        T_D_n = ds["T_D"][n],
        Constants = Constants)
    
    # C_a_n, C_l_n, T_s_n, Constants
    dC_l_n = C_l_tendency(
        C_a_n = ds["C_a"][n], 
        C_l_n = ds["C_l"][n], 
        T_s_n = ds["T_s"][n], 
        Constants = Constants)
    
    dC_s_n = C_s_tendency(
        C_a_n = ds["C_a"][n], 
        C_s_n = ds["C_s"][n], 
        C_d_n = ds["C_d"][n], 
        Constants = Constants)
    
    dC_d_n = C_d_tendency(
        C_s_n = ds["C_s"][n], 
        C_d_n = ds["C_d"][n], 
        Constants = Constants)

    dC_a_n = C_a_tendency(
        n = n, 
        dC_l_n = dC_l_n, 
        dC_s_n = dC_s_n, 
        dC_d_n = dC_d_n, 
        Constants = Constants,
        if_emission=True)

    # update to t=n+1
    ds["T_s"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["T_s"][n], 
        dF_n = dT_s_n)

    ds["T_D"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["T_D"][n], 
        dF_n = dT_D_n)
    
    ds["C_l"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["C_l"][n], 
        dF_n = dC_l_n)
    
    ds["C_s"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["C_s"][n], 
        dF_n = dC_s_n)
    
    ds["C_d"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["C_d"][n], 
        dF_n = dC_d_n)
    
    ds["C_a"][n+1] = euler_forward_func(
        dt = dt, 
        F_n = ds["C_a"][n], 
        dF_n = dC_a_n)
    # ds["C_a"][n+1] = Constants["CC_a_0"] * 2

    # ds["C_l"][n+1] =  dC_l_n
    # ds["C_s"][n+1] =  dC_s_n
    # ds["C_d"][n+1] =  dC_d_n
    # ds["C_a"][n+1] =  dC_a_n


# %% 
# Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
axes[0].plot(ds["T_s"], label="Surface Ocean T")
axes[0].plot(ds["T_D"], label="Deep Ocean T")
axes[0].set_xlabel("Time [years]")
axes[0].set_ylabel("K")
axes[0].legend()
axes[0].grid(linestyle="--", alpha=0.5)
axes[0].set_title("Temperature Anomaly [K]")
# plt.ylim([-10, 10])
# plt.xlim([0,25])

axes[1].plot(ds["C_s"], label="Surface Ocean")
axes[1].plot(ds["C_d"], label="Deep Ocean")
axes[1].plot(ds["C_a"], label="Atmospheric")
axes[1].plot(ds["C_l"], label="Land")
axes[1].set_xlabel("Time [years]")
axes[1].set_ylabel("GtC")
axes[1].legend()
axes[1].grid(linestyle="--", alpha=0.5)
axes[1].set_title("Carbon Anomaly [GtC]")
# plt.ylim([-10, 10])
# plt.xlim([0,25])
plt.show()
# %%

print(ds["C_l"])
print(ds["C_s"])
print(ds["C_d"])
print(ds["C_a"])

# %%
