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

X_LB = -2
X_UB = -.1
roi_x = slice(X_LB * D, X_UB * D)
roi_y = slice(-2 * D, 2 * D)

flow_roi = dat.sel(x=roi_x, y=roi_y)
target_x = flow_roi.x
target_y = flow_roi.y

TIs = np.arange(0.05, 0.45, 0.05)
WSs = np.arange(4, 11)
full_ti = [TIs for _ in range(WSs.size)]
full_ti = np.array(full_ti).flatten()
full_ws = [[WSs[ii]] * TIs.size for ii in range(WSs.size)]
full_ws = np.array(full_ws).flatten()
assert (full_ws.size == full_ti.size)

turbine = DTU10MW()
site = Hornsrev1Site()

obs_values = []
sim_res = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                            superpositionModel=LinearSum(), deflectionModel=None,
                            turbulenceModel=CrespoHernandez(),
                            blockage_deficitModel=SelfSimilarityDeficit2020(ss_alpha=0.2))([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True)

flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))

for t in range(flow_map.time.size):
    this_pred_sim = sim_res.isel(time=t, wt=0)
    observed_deficit = flow_roi.deficits.interp(ct=this_pred_sim.CT, ti=this_pred_sim.TI, z=0)
    obs_values.append(observed_deficit.T)

all_obs = xr.concat(obs_values, dim='time')

    #r12p=array([-0.672 ,  0.4897]),
    #ngp=array([-1.381,  2.627, -1.524,  1.336]),
    #fgp=array([-0.06489,  0.4911 ,  1.116  , -0.1577 ]),
def evaluate_rmse(ss_alpha, ss_beta, rp1, rp2, ng1, ng2, ng3, ng4, fg1, fg2, fg3, fg4):
    blockage_args = {'ss_alpha': ss_alpha, 'ss_beta': ss_beta, 'r12p': np.array([rp1, rp2]), 'ngp': np.array([ng1, ng2, ng3, ng4]),
                     'fgp': np.array([fg1, fg2, fg3, fg4])}

    wfm = All2AllIterative(site, turbine,
                                wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                                superpositionModel=LinearSum(), deflectionModel=None,
                                turbulenceModel=CrespoHernandez(),
                                blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args))

    sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True)
    flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))
    pred = (sim_res.WS - flow_map.WS_eff.isel(h=0)) / sim_res.WS

    rmse = float(np.sqrt(((all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
    if np.isnan(rmse): return -0.5
    return -rmse

pbounds = {
    'ss_alpha': (0.05, 3),
    'ss_beta': (0.05, 3),
    'rp1': (-2, 2),
    'rp2': (-2, 2),
    'ng1': (-3, 3),
    'ng2': (-3, 3),
    'ng3': (-3, 3),
    'ng4': (-3, 3),
    'fg1': (-2, 2),
    'fg2': (-2, 2),
    'fg3': (-2, 2),
    'fg4': (-2, 2)
}

#defaults = {'a_s': 0.17, 'b_s': 0.005, 'c_s': 0.2, 'b_f': -0.68, 'c_f': 2.41, 'ss_alpha': 0.8888888888888888, 'ss_beta': 1.4142135623730951}
defaults = {
    'ss_alpha': 0.8888888888888888,
    'ss_beta': 1.4142135623730951,
    'rp1': -0.672,
    'rp2': 0.4897,
    'ng1': -1.381,
    'ng2': 2.627,
    'ng3': -1.524,
    'ng4': 1.336,
    'fg1': -0.06489,
    'fg2': 0.4911,
    'fg3': 1.116,
    'fg4': -0.1577
    }

optimizer = BayesianOptimization(f=evaluate_rmse, pbounds=pbounds, random_state=1)
optimizer.probe(params=defaults, lazy=True)
optimizer.maximize(init_points=200, n_iter=200)

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
    keys = list(best_so_far_params.keys())
    best_vals = []
    default_vals = []
    for key in keys:
        best_vals.append(best_so_far_params[key])
        default_vals.append(defaults[key])



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

blockage_args = {'ss_alpha': best_params['ss_alpha'], 'ss_beta': best_params['ss_beta'], 'r12p': np.array([best_params['rp1'], best_params['rp2']]), 'ngp': np.array([best_params['ng1'], best_params['ng2'], best_params['ng3'], best_params['ng4']]),
                     'fgp': np.array([best_params['fg1'], best_params['fg2'], best_params['fg3'], best_params['fg4']])}

wfm = All2AllIterative(site, turbine,
                            wake_deficitModel=BlondelSuperGaussianDeficit2020(),
                            superpositionModel=LinearSum(), deflectionModel=None,
                            turbulenceModel=CrespoHernandez(),
                            blockage_deficitModel=SelfSimilarityDeficit2020(**blockage_args))

sim_res = wfm([0], [0], ws=full_ws, TI=full_ti, wd=[270] * full_ti.size, time=True)
flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))

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
    plt.savefig(f'figs/upstrem_err_{t}')
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
plt.title('Optimal RMSE: %.4f' % best_rmse)
plt.tight_layout()
plt.savefig('bar_LB_%i_UP_%i' % (X_LB, X_UB))
plt.clf()

