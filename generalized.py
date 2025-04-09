import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020
from py_wake import HorizontalGrid
from py_wake.deflection_models import JimenezWakeDeflection # Although not used (deflectionModel=None)
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter # Although not used
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.ground_models import Mirror
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake.deficit_models.utils import ct2a_mom1d
from bayes_opt import BayesianOptimization
import os # For creating figure directory

# --- Configuration ---
MODEL = 1  # 1: Blondel wake, 2: TurboGaussian wake
DOWNWIND = not True # True: Focus on wake region (2D to 10D), False: Focus on blockage (-2D to -1D)

# Create figure directory
FIG_DIR = f'figs_model_{MODEL}_downwind_{DOWNWIND}'
os.makedirs(FIG_DIR, exist_ok=True)

# --- Load Data and Turbine ---
print("Loading data and turbine...")
try:
    dat = xr.load_dataset('./DTU10MW.nc')
except FileNotFoundError:
    print("Error: DTU10MW.nc not found. Please ensure the file is in the correct directory.")
    exit()

turbine = DTU10MW()
D = turbine.diameter()
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)

# --- Define Region of Interest (ROI) ---
if DOWNWIND:
    X_LB, X_UB = 2, 10
    print(f"Analyzing DOWNWIND region: x = [{X_LB}D, {X_UB}D]")
else:
    X_LB, X_UB = -2, -1
    print(f"Analyzing UPWIND (blockage) region: x = [{X_LB}D, {X_UB}D]")

roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D) # Fixed y-range for both cases

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x.values
target_y = flow_roi.y.values
TARGET_GRID = HorizontalGrid(x=target_x, y=target_y) # Define grid once

# --- Define Simulation Conditions ---
print("Defining simulation conditions...")
TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)

# Create flattened arrays for WS and TI combinations
full_ti = np.repeat(TIs, len(WSs))
full_ws = np.tile(WSs, len(TIs))
assert full_ws.size == full_ti.size
N_CASES = full_ws.size
print(f"Total simulation cases: {N_CASES}")

# Site (used for ambient conditions, single turbine at [0,0])
site = Hornsrev1Site()
wt_x = [0]
wt_y = [0]
wd = [270] * N_CASES # Wind direction fixed for single turbine wake analysis

# --- Define Parameter Bounds and Defaults ---
print(f"Setting up parameters for MODEL={MODEL}, DOWNWIND={DOWNWIND}...")
if MODEL == 1: # BlondelSuperGaussianDeficit2020 focus
    if DOWNWIND:
        pbounds = {
            'a_s': (0.001, 0.5), 'b_s': (0.001, 0.01), 'c_s': (0.001, 0.5),
            'b_f': (-2, 1), 'c_f': (0.1, 5),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {
            'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41,
            'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32
        }
    else: # UPWIND - Focus on SelfSimilarityDeficit parameters
        pbounds = {
            'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
            'rp1': (-2, 2), 'rp2': (-2, 2),
            'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3),
            # Note: fg parameters seem unused for SelfSimilarityDeficit, maybe intended for future use?
            #'fg1': (-2, 2), 'fg2': (-2, 2), 'fg3': (-2, 2), 'fg4': (-2, 2)
        }
        defaults = {
            'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951,
            'rp1': -0.672, 'rp2': 0.4897,
            'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
            #'fg1': -0.06489, 'fg2': 0.4911, 'fg3': 1.116, 'fg4': -0.1577
        }
        # Need default CrespoHernandez params if optimizing blockage
        defaults.update({'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32})
        pbounds.update({'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2)})


elif MODEL == 2: # TurboGaussianDeficit focus (only DOWNWIND defined in original code)
    if DOWNWIND:
        pbounds = {
            'A': (0.001, .5), 'cti1': (.01, 5), 'cti2': (0.01, 5),
            'ceps': (0.01, 3), 'ctlim': (0.01, 1),
            'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2),
        }
        defaults = {
            'A': 0.04, 'cti1': 1.5, 'cti2': 0.8, 'ceps': 0.25, 'ctlim': 0.999,
            'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32
        }
    else: # UPWIND - Focus on SelfSimilarityDeficit parameters
         # Using same bounds/defaults as MODEL=1 UPWIND for blockage part
        pbounds = {
            'ss_alpha': (0.05, 3), 'ss_beta': (0.05, 3),
            'rp1': (-2, 2), 'rp2': (-2, 2),
            'ng1': (-3, 3), 'ng2': (-3, 3), 'ng3': (-3, 3), 'ng4': (-3, 3),
        }
        defaults = {
            'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951,
            'rp1': -0.672, 'rp2': 0.4897,
            'ng1': -1.381, 'ng2': 2.627, 'ng3': -1.524, 'ng4': 1.336,
        }
        # Need default CrespoHernandez params if optimizing blockage
        defaults.update({'ch1': 0.73, 'ch2': 0.8325, 'ch3': -0.0325, 'ch4': -0.32})
        pbounds.update({'ch1': (-1, 2), 'ch2': (-1, 2), 'ch3': (-1, 2), 'ch4': (-1, 2)})

else:
    raise ValueError(f"Invalid MODEL: {MODEL}. Choose 1 or 2.")

# --- Modular Wind Farm Model Creation ---
def create_wfm(params, site_model, turbine_model, model_id, is_downwind):
    """Creates the PyWake WindFarmModel based on flags and parameters."""

    wake_deficitModel = None
    blockage_deficitModel = None
    turbulenceModel = None

    # Common models
    superpositionModel = LinearSum()
    deflectionModel = None # Not optimizing deflection

    # Parameter subsets
    ch_keys = ['ch1', 'ch2', 'ch3', 'ch4']
    ss_keys = ['ss_alpha', 'ss_beta', 'r12p', 'ngp'] # Simplified for SelfSimilarity
    blondel_keys = ['a_s', 'b_s', 'c_s', 'b_f', 'c_f']
    turbo_keys = ['A', 'cti1', 'cti2', 'ceps', 'ctlim']

    # --- Configure Turbulence Model (always CrespoHernandez for now) ---
    turb_args = {'c': np.array([params.get(k, defaults.get(k)) for k in ch_keys])} # Use default if key missing
    turbulenceModel = CrespoHernandez(**turb_args)

    # --- Configure Wake/Blockage Models based on flags ---
    if is_downwind:
        # Focus on WAKE model, use default blockage
        blockage_deficitModel = SelfSimilarityDeficit2020(groundModel=Mirror()) # Default SS params, Mirror ground

        if model_id == 1:
            def_args = {k: params[k] for k in blondel_keys}
            wake_deficitModel = BlondelSuperGaussianDeficit2020(**def_args)
        elif model_id == 2:
            def_args = {
                'A': params['A'],
                'cTI': [params['cti1'], params['cti2']],
                'ceps': params['ceps'],
                'ctlim': params['ctlim'],
                'ct2a': ct2a_mom1d, # Kept from original
                'groundModel': Mirror(),
                'rotorAvgModel': GaussianOverlapAvgModel()
            }
            wake_deficitModel = TurboGaussianDeficit(**def_args)
            # wake_deficitModel.WS_key = 'WS_jlk' # From original, check if needed - usually handled internally
        else:
             raise ValueError(f"Invalid model_id: {model_id}")

    else: # UPWIND - Focus on BLOCKAGE model, use default wake
        # Use default wake model based on model_id
        if model_id == 1:
             wake_deficitModel = BlondelSuperGaussianDeficit2020() # Default Blondel params
        elif model_id == 2:
             # Need a default TurboGaussian configuration if MODEL=2 is used for blockage opt
             wake_deficitModel = TurboGaussianDeficit(groundModel=Mirror(), rotorAvgModel=GaussianOverlapAvgModel()) # Default Turbo params
        else:
             raise ValueError(f"Invalid model_id: {model_id}")

        # Configure Blockage model with optimized parameters
        blockage_args = {
            'ss_alpha': params['ss_alpha'],
            'ss_beta': params['ss_beta'],
            'r12p': np.array([params.get('rp1', defaults['rp1']), params.get('rp2', defaults['rp2'])]), # Handle potential missing keys if bounds change
            'ngp': np.array([params.get(f'ng{i}', defaults[f'ng{i}']) for i in range(1, 5)])
        }
        # Add ground model based on original logic (Mirror only if MODEL=2?) - Let's add it always for SelfSim blockage
        blockage_args['groundModel'] = Mirror()
        blockage_deficitModel = SelfSimilarityDeficit2020(**blockage_args)


    # --- Instantiate WindFarmModel ---
    wfm = All2AllIterative(
        site_model,
        turbine_model,
        wake_deficitModel=wake_deficitModel,
        superpositionModel=superpositionModel,
        deflectionModel=deflectionModel,
        turbulenceModel=turbulenceModel,
        blockage_deficitModel=blockage_deficitModel
    )
    return wfm

# --- Simulation and Deficit Calculation ---

def run_and_calc_deficit(wfm, grid, ws, ti, wd, x, y):
    """
    Runs the simulation and calculates the predicted deficit field by
    computing the flow map one case at a time.

    Args:
        wfm: Instantiated PyWake WindFarmModel.
        grid: py_wake.HorizontalGrid object for flow map calculation.
        ws (array_like): Array of wind speeds for cases.
        ti (array_like): Array of turbulence intensities for cases.
        wd (array_like): Array of wind directions for cases.
        x (array_like): Wind turbine x-coordinates.
        y (array_like): Wind turbine y-coordinates.

    Returns:
        tuple: (sim_res, pred_deficit)
            sim_res (xarray.Dataset): The full simulation result object.
            pred_deficit (xarray.DataArray): Predicted deficit field with
                                             dimensions ('case', 'y', 'x').
                                             Returns None if simulation fails.
    """
    print(f"Running simulation for {len(ws)} cases...")
    # Run the simulation for all cases at once to get sim_res
    # This is efficient for WT-local results like CT, Power etc.
    # Let potential errors during the simulation itself propagate here.
    sim_res = wfm(x, y, ws=ws, TI=ti, wd=wd, time=True)

    n_cases = len(ws) # Number of simulation cases

    # Calculate flow map one time step (case) at a time
    ws_eff_list = []
    for t in range(n_cases):
        # Let potential errors during flow_map calculation propagate
        # If flow_map fails for case 't', the script will stop here.
        flow_map_t = sim_res.flow_map(grid, time=[t])

        # Append the WS_eff part. flow_map_t.WS_eff should have dims (time=1, h, y, x)
        ws_eff_list.append(flow_map_t.WS_eff)
    print("\nFlow map calculation completed.")

    # Concatenate the list of single-time WS_eff results along the 'time' dimension
    # This reconstructs the full WS_eff field corresponding to all cases
    all_ws_eff = xr.concat(ws_eff_list, dim='time') # Dims should be (time, h, y, x)

    # Select the hub height wind speed. Assuming index h=0 is the desired hub height.
    # If multiple heights were simulated, ensure this selection is correct.
    hub_ws_eff = all_ws_eff.isel(h=0) # Dims (time, y, x)

    # Prepare the freestream wind speed array for deficit calculation
    # Ensure sim_res.WS aligns with hub_ws_eff (time dimension)
    # If x, y represent a single turbine [0], [0], sim_res.WS likely has dims (time, wt)
    ws_freestream_xr = xr.DataArray(ws, coords={'time': sim_res.time}, dims=['time'])

    # Alternative: Use sim_res.WS if inflow is uniform and only one turbine considered in deficit
    # elif 'wt' in sim_res.WS.dims and sim_res.WS.sizes['wt'] == 1:
    #     ws_freestream_xr = sim_res.WS.isel(wt=0) # Dims (time,)
    # elif 'time' in sim_res.WS.dims and 'wt' not in sim_res.WS.dims:
    #     ws_freestream_xr = sim_res.WS # Dims (time,)



    # Calculate the deficit (WS_free - WS_eff) / WS_free
    # xarray handles broadcasting ws_freestream_xr (time,) to (time, y, x)
    pred_deficit = (ws_freestream_xr - hub_ws_eff) / ws_freestream_xr

    # Rename the 'time' dimension to 'case' for consistency with downstream processing (like all_obs)
    pred_deficit = pred_deficit.rename({'time': 'case'})

    return sim_res, pred_deficit


# --- Prepare Target Deficits (Interpolate ONCE) ---
print("Preparing target deficit data...")
# Run simulation once with DEFAULT parameters to get corresponding CT/TI for interpolation
wfm_default_init = create_wfm(defaults, site, turbine, MODEL, DOWNWIND)
sim_res_default_init, _ = run_and_calc_deficit(wfm_default_init, TARGET_GRID, full_ws, full_ti, wd, wt_x, wt_y) # No need for deficit here

obs_values = []
for i in range(N_CASES):
    # Get CT and TI for this specific case from the initial default simulation
    ct_i = sim_res_default_init.CT.isel(time=i).item()
    ti_i = sim_res_default_init.TI.isel(time=i).item()
    # Interpolate the reference data for this CT/TI
    # Using fill_value=None to let xarray handle edges, may result in NaNs
    observed_deficit_i = flow_roi.deficits.interp(ct=ct_i, ti=ti_i, z=0, method='linear', kwargs={'fill_value': None})
    obs_values.append(observed_deficit_i)

# Concatenate along a new 'case' dimension
all_obs = xr.concat(obs_values, dim='case').assign_coords(case=np.arange(N_CASES))
# Transpose to match predicted deficit dimensions (case, y, x) - Check if needed after concat
if all_obs.dims != ('case', 'y', 'x'):
     # Assuming original dims might be (case, x, y) based on previous .T usage
     if all_obs.dims == ('case', 'x', 'y'):
         all_obs = all_obs.transpose('case', 'y', 'x')
     else:
         print(f"Warning: Unexpected all_obs dimensions: {all_obs.dims}. Required ('case', 'y', 'x')")

print(f"Target deficits shape: {all_obs.shape}") # Should be (N_CASES, Ny, Nx)

# --- Objective Function for Bayesian Optimization ---
def evaluate_rmse(**kwargs):
    """Objective function: calculates RMSE between simulation and target deficits."""
    # 1. Create WFM with current parameters
    wfm_opt = create_wfm(kwargs, site, turbine, MODEL, DOWNWIND)

    # 2. Run simulation and calculate predicted deficit
    _, pred_deficit = run_and_calc_deficit(wfm_opt, TARGET_GRID, full_ws, full_ti, wd, wt_x, wt_y)

    # 3. Calculate error and RMSE
    error = all_obs - pred_deficit
    # Calculate RMSE over spatial dimensions (y, x) first, then average over cases
    # Handle potential NaNs from interpolation or simulation errors
    rmse_per_case = np.sqrt(((error**2).mean(dim=['x', 'y'])))
    mean_rmse = float(rmse_per_case.mean(skipna=True)) # Average RMSE across all cases

    # Penalty for NaNs - adjust penalty value as needed
    if np.isnan(mean_rmse) or not np.isfinite(mean_rmse):
        print(f"Warning: NaN RMSE encountered for params: {kwargs}. Returning large penalty.")
        return -1.0 # Return a large negative value (bad score)

    # BayesOpt maximizes, so return negative RMSE
    return -mean_rmse

# --- Bayesian Optimization ---
print("Starting Bayesian Optimization...")
optimizer = BayesianOptimization(
    f=evaluate_rmse,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True # Might be useful if parameter space is tricky
)

# Probe default parameters first
print("Probing default parameters...")
optimizer.probe(params=defaults, lazy=True)

# Run optimization
optimizer.maximize(init_points=50, n_iter=200) # Use values from original script

best_params = optimizer.max['params']
best_neg_rmse = optimizer.max['target']
print(f"Optimization finished. Best params found: {best_params}")
print(f"Best RMSE achieved: {-best_neg_rmse:.6f}")


# --- Generate Optimization Animation ---
print("Generating optimization animation...")
def update_plot_animation(frame, optimizer_instance, default_params):
    ax1 = fig_ani.axes[0]
    ax2 = fig_ani.axes[1]
    ax1.clear()
    ax2.clear()

    # Get optimization history up to current frame
    targets = -np.array(optimizer_instance.space.target[:frame+1]) # Convert back to positive RMSE
    best_historical_rmse = np.minimum.accumulate(targets)
    best_iteration_index = np.argmin(targets) # Index of the best result in this frame's history
    current_best_params = optimizer_instance.res[best_iteration_index]['params']
    current_best_rmse = targets[best_iteration_index]

    # Plot RMSE convergence
    ax1.plot(targets, color='gray', alpha=0.5, label='RMSE per Iteration')
    ax1.plot(best_historical_rmse, color='black', label='Best RMSE Found')
    ax1.set_title('Optimization Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)
    ax1.legend()

    # Bar plot for parameters (comparing current best vs default)
    keys = list(default_params.keys())
    # Ensure current_best_params has all keys from defaults for comparison
    best_vals = [current_best_params.get(k, np.nan) for k in keys] # Use NaN if key missing
    default_vals = [default_params[k] for k in keys]
    x_pos = np.arange(len(keys))

    ax2.bar(x_pos - 0.2, best_vals, width=0.4, label=f'Best (Iter {best_iteration_index})')
    ax2.bar(x_pos + 0.2, default_vals, width=0.4, label='Default')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(keys, rotation=45, ha="right")
    ax2.set_title(f'Parameters (Best RMSE: {current_best_rmse:.4f})')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--')
    fig_ani.tight_layout() # Apply tight layout within the update function

fig_ani, (ax_ani1, ax_ani2) = plt.subplots(1, 2, figsize=(15, 6))
ani = animation.FuncAnimation(fig_ani, update_plot_animation,
                              frames=len(optimizer.space.target), # Number of iterations performed
                              fargs=(optimizer, defaults), # Pass optimizer and defaults
                              repeat=False)

# Save as MP4
try:
    ani_filename = os.path.join(FIG_DIR, f'optimization_animation_LB_{X_LB}_UB_{X_UB}.mp4')
    writer = animation.FFMpegWriter(fps=15)
    ani.save(ani_filename, writer=writer)
    print(f"Animation saved to {ani_filename}")
except FileNotFoundError:
    print("Error: FFmpeg not found. Cannot save animation. Please install FFmpeg.")
except Exception as e:
    print(f"Error saving animation: {e}")
plt.close(fig_ani)


# --- Evaluate and Compare Default vs Optimized Performance ---
print("Evaluating final performance: Default vs Optimized...")

# 1. Default Performance
print(" - Running default simulation...")
wfm_default = create_wfm(defaults, site, turbine, MODEL, DOWNWIND)
sim_res_default, pred_deficit_default = run_and_calc_deficit(wfm_default, TARGET_GRID, full_ws, full_ti, wd, wt_x, wt_y)
error_default = all_obs - pred_deficit_default
rmse_default_overall = float(np.sqrt(((error_default**2).mean(skipna=True))))
mean_error_default = error_default.mean(dim='case', skipna=True)
std_dev_error_default = error_default.std(dim='case', skipna=True)
p90_abs_error_default = abs(error_default).quantile(0.9, dim='case', skipna=True)
print(f"   Default Overall RMSE: {rmse_default_overall:.6f}")

# 2. Optimized Performance
print(" - Running optimized simulation...")
wfm_optimized = create_wfm(best_params, site, turbine, MODEL, DOWNWIND)
sim_res_optimized, pred_deficit_optimized = run_and_calc_deficit(wfm_optimized, TARGET_GRID, full_ws, full_ti, wd, wt_x, wt_y)
error_optimized = all_obs - pred_deficit_optimized
rmse_optimized_overall = float(np.sqrt(((error_optimized**2).mean(skipna=True))))
mean_error_optimized = error_optimized.mean(dim='case', skipna=True)
std_dev_error_optimized = error_optimized.std(dim='case', skipna=True)
p90_abs_error_optimized = abs(error_optimized).quantile(0.9, dim='case', skipna=True)
print(f"   Optimized Overall RMSE: {rmse_optimized_overall:.6f}")


# --- Generate Summary Comparison Plots ---
print("Generating summary comparison plots...")

# Determine shared colorbar limits for error plots
max_abs_mean_err = max(abs(mean_error_default).max(), abs(mean_error_optimized).max())
vmax_std = max(std_dev_error_default.max(), std_dev_error_optimized.max())
vmax_p90 = max(p90_abs_error_default.max(), p90_abs_error_optimized.max())

fig_comp, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
fig_comp.suptitle(f'Error Comparison (Model {MODEL}, {"Downwind" if DOWNWIND else "Upwind"})', fontsize=16)

# Plot Mean Error
im00 = axes[0, 0].contourf(target_x/D, target_y/D, mean_error_default, levels=20, cmap='coolwarm', vmin=-max_abs_mean_err, vmax=max_abs_mean_err)
axes[0, 0].set_title(f'Default Mean Error (RMSE: {rmse_default_overall:.4f})')
fig_comp.colorbar(im00, ax=axes[0, 0], label='Mean Deficit Error')

im10 = axes[1, 0].contourf(target_x/D, target_y/D, mean_error_optimized, levels=20, cmap='coolwarm', vmin=-max_abs_mean_err, vmax=max_abs_mean_err)
axes[1, 0].set_title(f'Optimized Mean Error (RMSE: {rmse_optimized_overall:.4f})')
fig_comp.colorbar(im10, ax=axes[1, 0], label='Mean Deficit Error')

# Plot Std Dev Error
im01 = axes[0, 1].contourf(target_x/D, target_y/D, std_dev_error_default, levels=20, cmap='viridis', vmin=0, vmax=vmax_std)
axes[0, 1].set_title('Default Std Dev Error')
fig_comp.colorbar(im01, ax=axes[0, 1], label='Std Dev Deficit Error')

im11 = axes[1, 1].contourf(target_x/D, target_y/D, std_dev_error_optimized, levels=20, cmap='viridis', vmin=0, vmax=vmax_std)
axes[1, 1].set_title('Optimized Std Dev Error')
fig_comp.colorbar(im11, ax=axes[1, 1], label='Std Dev Deficit Error')

# Plot P90 Absolute Error
im02 = axes[0, 2].contourf(target_x/D, target_y/D, p90_abs_error_default, levels=20, cmap='magma', vmin=0, vmax=vmax_p90)
axes[0, 2].set_title('Default P90 Abs Error')
fig_comp.colorbar(im02, ax=axes[0, 2], label='P90 Abs Deficit Error')

im12 = axes[1, 2].contourf(target_x/D, target_y/D, p90_abs_error_optimized, levels=20, cmap='magma', vmin=0, vmax=vmax_p90)
axes[1, 2].set_title('Optimized P90 Abs Error')
fig_comp.colorbar(im12, ax=axes[1, 2], label='P90 Abs Deficit Error')

# Add labels
axes[0,0].set_ylabel('Default\ny/D')
axes[1,0].set_ylabel('Optimized\ny/D')
for ax in axes[1, :]:
    ax.set_xlabel('x/D')

plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
comp_filename = os.path.join(FIG_DIR, f'error_comparison_summary_LB_{X_LB}_UP_{X_UB}.png')
plt.savefig(comp_filename)
print(f"Summary comparison plot saved to {comp_filename}")
plt.close(fig_comp)


# --- Final Parameter Bar Plot ---
print("Generating final parameter comparison bar plot...")
fig_bar, ax_bar = plt.subplots(figsize=(max(10, len(defaults) * 0.8), 6)) # Adjust width based on num params

keys = list(defaults.keys())
best_vals_final = [best_params.get(k, np.nan) for k in keys] # Use NaN if param wasn't optimized (e.g., fixed CH params when optimizing SS)
default_vals_final = [defaults[k] for k in keys]
x_pos = np.arange(len(keys))

width = 0.35
rects1 = ax_bar.bar(x_pos - width/2, best_vals_final, width, label=f'Optimized (RMSE: {rmse_optimized_overall:.4f})')
rects2 = ax_bar.bar(x_pos + width/2, default_vals_final, width, label=f'Default (RMSE: {rmse_default_overall:.4f})')

ax_bar.set_ylabel('Parameter Value')
ax_bar.set_title(f'Optimized vs Default Parameters (Model {MODEL}, {"Downwind" if DOWNWIND else "Upwind"})')
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(keys, rotation=45, ha="right")
ax_bar.legend()
ax_bar.grid(axis='y', linestyle='--')

# Add value labels on bars (optional, can be cluttered)
# ax_bar.bar_label(rects1, padding=3, fmt='%.2f', rotation=90)
# ax_bar.bar_label(rects2, padding=3, fmt='%.2f', rotation=90)

plt.tight_layout()
bar_filename = os.path.join(FIG_DIR, f'param_comparison_bar_LB_{X_LB}_UP_{X_UB}.png')
plt.savefig(bar_filename)
print(f"Parameter comparison plot saved to {bar_filename}")
plt.close(fig_bar)

print("\n--- Script Finished ---")
print("BEST Parameters:")
print(best_params)
