import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020
from py_wake import HorizontalGrid
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.turbulence_models import CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter
from py_wake.deficit_models import SelfSimilarityDeficit2020
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.superposition_models import LinearSum
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from bayes_opt import BayesianOptimization

dat = xr.load_dataset('./DTU10MW.nc')
D = 178.3
dat = dat.assign_coords(x=dat.x * D, y=dat.y * D)

X_LB = 2
X_UB = 5
roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D)

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x
target_y = flow_roi.y

turbine = DTU10MW()
site = Hornsrev1Site()

obs_values = []
sim_res = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                            superpositionModel=LinearSum(), deflectionModel=None,
                            turbulenceModel=CrespoHernandez(),
                            blockage_deficitModel=SelfSimilarityDeficit2020(ss_alpha=0.2))([0], [0], ws=[6, 6, 10, 10], TI=[0.1, 0.4, 0.1, 0.4], wd=[270] * 4, time=True)

flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))

for t in range(flow_map.time.size):
    this_pred_sim = sim_res.isel(time=t, wt=0)
    observed_deficit = flow_roi.deficits.interp(ct=this_pred_sim.CT, ti=this_pred_sim.TI, z=0)
    obs_values.append(observed_deficit.T)

all_obs = xr.concat(obs_values, dim='time')

def evaluate_rmse(a_s, b_s, c_s, b_f, c_f, ss_alpha, ss_beta):
    def_args = {'a_s': a_s, 'b_s': b_s, 'c_s': c_s, 'b_f': b_f, 'c_f': c_f, 'rotorAvgModel': RotorCenter(), 'groundModel': None}
    blockage_args = {'ss_alpha': ss_alpha, 'ss_beta': ss_beta}

    wfm = All2AllIterative(site, turbine,
                                wake_deficitModel=BlondelSuperGaussianDeficit2020(**def_args),
                                superpositionModel=LinearSum(), deflectionModel=None,
                                turbulenceModel=CrespoHernandez(),
                                blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args))

    sim_res = wfm([0], [0], ws=[6, 6, 10, 10], TI=[0.1, 0.4, 0.1, 0.4], wd=[270] * 4, time=True)
    flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))
    pred = (sim_res.WS - flow_map.WS_eff.isel(h=0)) / sim_res.WS

    rmse = float(np.sqrt(((all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
    if np.isnan(rmse): return -0.5
    return -rmse

pbounds = {
    'a_s': (0.001, 0.5),
    'b_s': (0.001, 0.01),
    'c_s': (0.001, 0.5),
    'b_f': (-2, 1),
    'c_f': (0.1, 5),
    'ss_alpha': (0.05, 3),
    'ss_beta': (0.05, 3)
}

defaults = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41, 'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951}

optimizer = BayesianOptimization(f=evaluate_rmse, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=50, n_iter=200)

best_params = optimizer.max['params']
best_rmse = -optimizer.max['target']

def update_plot(frame):
    ax1.clear()
    ax2.clear()

    # Get the best parameters and corresponding RMSE up to the current frame
    best_so_far_params = {}
    best_so_far_rmse = float('inf')
    best_so_far_rmses = []
    for i in range(frame + 1):
        if -optimizer.space.target[i] <= best_so_far_rmse:
            best_so_far_rmse = -optimizer.space.target[i]
            best_so_far_params = optimizer.res[i]['params']
        best_so_far_rmses.append(best_so_far_rmse)

    # Plot the entire history in gray
    ax1.plot(-np.array(optimizer.space.target), color='gray', alpha=0.5)
    # Plot the best RMSE so far in black
    ax1.plot(np.array(best_so_far_rmses), color='black')
    ax1.set_title('Optimization Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)

    # Use the best parameters so far for the bar plot
    best_vals = list(best_so_far_params.values())
    default_vals = list(defaults.values())
    keys = list(best_so_far_params.keys())

    ax2.bar(keys, best_vals, label='Optimized')
    ax2.bar(keys, default_vals, edgecolor='black', linewidth=2, color='none', capstyle='butt', label='Default')
    ax2.set_title(f'Best RMSE: {best_so_far_rmse:.4f}')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    plt.tight_layout()
    return ax1, ax2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ani = animation.FuncAnimation(fig, update_plot, frames=len(optimizer.space.target), repeat=False)

# Save as MP4
writer = animation.FFMpegWriter(fps=15)  # You can adjust fps as needed
ani.save('optimization_animation_%i_%i.mp4' % (X_LB, X_UB), writer=writer)
plt.close('all')

def_args = {'a_s': best_params['a_s'], 'b_s': best_params['b_s'], 'c_s': best_params['c_s'], 'b_f': best_params['b_f'], 'c_f': best_params['c_f'], 'rotorAvgModel': RotorCenter(), 'groundModel': None}

wfm = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(**def_args),
                            superpositionModel=LinearSum(), deflectionModel=None,
                            turbulenceModel=CrespoHernandez(),
                            blockage_deficitModel=SelfSimilarityDeficit2020(ss_alpha=0.2))

sim_res = wfm([0], [0], ws=[6, 6, 10, 10], TI=[0.1, 0.4, 0.1, 0.4], wd=[270] * 4, time=True)

rmse_values = []
for t in range(flow_map.time.size):
    this_pred_sim = sim_res.isel(time=t)
    observed_deficit = flow_roi.deficits.interp(ct=this_pred_sim.CT, ti=this_pred_sim.TI, z=0).isel(wt=0)
    pred = (this_pred_sim.WS - flow_map.WS_eff.isel(h=0, time=t)) / this_pred_sim.WS
    diff = observed_deficit.T - pred
    rmse = np.sqrt(np.mean(diff**2))
    rmse_values.append(rmse)
    fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    co = ax[0].contourf(target_x, target_y, observed_deficit.T)
    cp = ax[1].contourf(target_x, target_y, pred)
    cd = ax[2].contourf(target_x, target_y, diff)
    for jj, c in enumerate([co, cp, cd]):
        fig.colorbar(c, ax=ax[jj])
    ax[0].set_ylabel('Observed')
    ax[1].set_ylabel('Prediction')
    ax[2].set_ylabel('Diff')
    plt.tight_layout()
    plt.savefig(f'figs/err_{t}')
    plt.clf()

overall_rmse = np.mean(rmse_values)
print(f"RMSE values per time step: {rmse_values}")
print(f"Overall RMSE: {overall_rmse}")

plt.close('all')

best_vals = []
default_vals = []
keys = best_params.keys()
for key in keys:
    best_vals.append(best_params[key])
    default_vals.append(defaults[key])

plt.bar(keys, best_vals)
plt.bar(keys, default_vals,
          edgecolor='black',    # Only the edges will be visible
          linewidth=2,          # Make the line thicker
          color='none',         # Make the bar fill transparent
          capstyle='butt')
plt.title('Optimal RMSE: %.4f' % overall_rmse)
plt.tight_layout()
plt.savefig('bar_LB_%i_UP_%i' % (X_LB, X_UB))
plt.clf()

