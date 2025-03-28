import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Define region of interest: x in [0, 10D] and y in [-2D, 2D]
roi_x = slice(2 * D, 10*D)
roi_y = slice(-2*D, 2*D)

# Subset flow_map to the region of interest
flow_roi = dat.sel(x=roi_x, y=roi_y)

# Extract x and y coordinates from flow_map ROI for interpolation
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

def evaluate_rmse(a_s, b_s, c_s, b_f, c_f):
    def_args = {'a_s': a_s, 'b_s': b_s, 'c_s': c_s, 'b_f': b_f, 'c_f': c_f, 'rotorAvgModel': RotorCenter(), 'groundModel': None}

    wfm = All2AllIterative(site, turbine,
                          wake_deficitModel=BlondelSuperGaussianDeficit2020(**def_args),
                          superpositionModel=LinearSum(), deflectionModel=None,
                          turbulenceModel=CrespoHernandez(),
                          blockage_deficitModel=SelfSimilarityDeficit2020(ss_alpha=0.2))

    sim_res = wfm([0], [0], ws=[6, 6, 10, 10], TI=[0.1, 0.4, 0.1, 0.4], wd=[270] * 4, time=True)
    flow_map = sim_res.flow_map(HorizontalGrid(x=target_x, y=target_y))
    pred = (sim_res.WS - flow_map.WS_eff.isel(h=0)) / sim_res.WS

    rmse = float(np.sqrt(((all_obs - pred) ** 2).mean(['x', 'y'])).mean('time'))
    return -rmse  # Bayesian optimization maximizes, so we negate RMSE

pbounds = {
    'a_s': (0.01, 0.5),
    'b_s': (0.001, 0.01),
    'c_s': (0.01, 0.5),
    'b_f': (0.1, 1),
    'c_f': (0.1, 5)
}

optimizer = BayesianOptimization(
    f=evaluate_rmse,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=50,
)

print(optimizer.max)

# Plot optimization convergence
plt.figure(figsize=(10, 6))
plt.plot(-optimizer.space.target)  # Negate to plot RMSE directly
plt.title('Optimization Convergence')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('optimization_convergence.png')
plt.clf()

best_params = optimizer.max['params']
best_rmse = -optimizer.max['target'] # reverse the negation.

print("Best Parameters:")
print(best_params)
print(f"Best RMSE: {best_rmse}")

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

plt.bar(best_params.keys(), best_params.values())
plt.title('Optimal RMSE: %.4f' % overall_rmse)
plt.tight_layout()
plt.savefig('bar_LB_%i_UP_%i')
plt.clf()
