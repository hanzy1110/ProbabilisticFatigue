import os
import json
import pathlib
import fire
import pymc as pm
import nutpie
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import collections, itertools
import scienceplots
from time import perf_counter

import jax.numpy as jnp

from scipy import stats
from pymc.distributions import Interpolated

from multiprocessing import Pool

plt.style.use(["science", "ieee"])

BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"
LOAD_PATH = RESULTS_FOLDER / "LOADS"
MAX_SAMPLES = 15000

N_YEARS = 100
N_BATCHES = 100


n_min = 1
n_max = 100
n_batches = 5

def print_perf(op, s,e):
    print(f"TIME SPENT ON {op}, {e-s} ")

def getPFailure(damages):
    exceeding = damages[np.where(damages>1)]
    print(f"SHAPES -> E: {exceeding.shape} D: {damages.shape}")
    print(f"EXCEEDING => {len(exceeding)}")
    print(f"TOTAL => {len(damages)}")
    # print(f"damages => {damages[:500]}")

    return exceeding.shape[0] / damages.shape[0]


def getVarCoeff(p_failures, N_mcs):
    return np.sqrt((1 - p_failures) / (N_mcs * p_failures))


def window(it, winsize, step=2):
    """Sliding window iterator."""
    it = iter(it)  # Ensure we have an iterator
    l = collections.deque(itertools.islice(it, winsize))
    while 1:  # Continue till StopIteration gets raised.
        try:
            yield tuple(l)
            for i in range(step):
                l.append(next(it))
                l.popleft()
        except StopIteration as e:
            return


def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    # x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    x = np.concatenate([[x[0] - 4 * width], x, [x[-1] + 4 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)


def get_prev_damage(year_init, year_end, dmg_model):
    prev_year_end = year_init + 1
    prev_year_init = prev_year_end - (year_end - year_init)
    ppc_filename = (
        RESULTS_FOLDER / f"damage_posterior_{prev_year_init}_{prev_year_end}_{dmg_model}.nc"
    )

    print(f"PPC FILENAME ==> {ppc_filename}")

    d_name = f"damage_{year_init}-{year_init-1}"
    prev_ppc = az.from_netcdf(ppc_filename)

    prev_damage_samples = prev_ppc.posterior_predictive[d_name].mean(
        dim=("chain", "draw")
    )

    return from_posterior(d_name, prev_damage_samples)


def get_tot_damages(year, dmg_model) -> np.ndarray:
    tot_damages = None
    for i in range(N_BATCHES):
        # print(f"BATCH NUMBER : {i}")

        try:
            with open(
                LOAD_PATH / f"tot_damages_year{year}_batch_{i}_{dmg_model}.npz", "rb"
            ) as file:
                samples = np.load(file, allow_pickle=True)
                # damages_aeran = {key: jnp.array(val) for key, val in samples.items()}
                # damages_aeran = jnp.array(samples['arr_0'])
                damages_aeran = samples["arr_0"]
                if tot_damages is not None:
                    tot_damages = np.hstack([tot_damages, damages_aeran])
                else:
                    tot_damages = damages_aeran
        except Exception as e:
            continue
            # print(e)

    if tot_damages is not None:
        # print("NAN FRAC ==>")
        # print(f"NAN LEN {len(tot_damages[np.isnan(tot_damages)])}")
        # print(len(tot_damages[np.isnan(tot_damages)]) / len(tot_damages))
        # print(f"TOTAL LEN {len(tot_damages)}")
        return tot_damages[~np.isnan(tot_damages)][:MAX_SAMPLES]

    return tot_damages


def build_damage_model(year_init, year_end, dmg_model="Aeran"):
    year_range = range(year_init, year_end)
    damages = [get_tot_damages(i, dmg_model) for i in year_range]
    tot_damages = np.array([d for d in damages if d is not None], dtype=np.float32)

    # with pm.Model(coords=coords) as damage_model:
    with pm.Model() as damage_model:
        for j, i in enumerate(year_range):
            alpha = pm.Gamma(f"alpha_{i}", alpha=1, beta=1)
            beta = pm.Gamma(f"beta_{i}", alpha=1, beta=1)
            damages = pm.Normal(
                f"damage_{i}",
                mu=alpha,
                sigma=beta,
                observed=tot_damages[j, :],
            )

    return damage_model


def sample_model(damage_model, draws, year_init, year_end, dmg_model="Aeran"):
    filename = RESULTS_FOLDER / f"DAMAGE_MODEL_TRACE_{year_init}_{year_end}_{dmg_model}.nc"

    if not os.path.exists(filename):
        compiled_model = nutpie.compile_pymc_model(damage_model)
        trace = nutpie.sample(compiled_model, draws=draws, tune=500, chains=2)
        az.to_netcdf(
            data=trace,
            filename=filename,
        )
    else:
        print("LOADING TRACE")
        trace = az.from_netcdf(filename)

    return trace


def posterior_sample(damage_model, trace, year_init, year_end, dmg_model="Aeran"):
    partial_year_range = range(year_init + 1, year_end)
    year_range = range(year_init, year_end)
    partial_names = [f"damage_{i}-{i-1}" for i in partial_year_range]
    names = [f"damage_{i}" for i in year_range]

    print(names)
    print(partial_names)

    ppc_filename = RESULTS_FOLDER / f"damage_posterior_{year_init}_{year_end}_{dmg_model}.nc"

    if not os.path.exists(ppc_filename):
        with damage_model:
            for i, n in enumerate(names[1:]):
                if i == 0:
                    if year_init != 0:
                        damage_prev = get_prev_damage(year_init, year_end, dmg_model)
                    else:
                        damage_prev = damage_model.named_vars[names[0]]
                else:
                    damage_prev = damage_model.named_vars[partial_names[i - 1]]
                # name = f"damage_{i}-{i-1}"
                # names.append(name)
                damage = damage_model.named_vars.get(n)
                d = pm.Deterministic(partial_names[i], damage + damage_prev)
            ppc = pm.sample_posterior_predictive(trace, var_names=partial_names)
            az.to_netcdf(ppc, ppc_filename)
    else:
        print("LOADING PPC")
        ppc = az.from_netcdf(ppc_filename)

    return ppc, partial_names


def post_process(ppc, n, plot, dmg_model="Aeran"):
    d = ppc.posterior_predictive[n]

    d_mean = d.mean(dim=("chain", "draw"))
    N_mcs = len(d_mean)
    p_failures = getPFailure(d_mean.values)
    # v_coeffs = getVarCoeff(d_mean.values, N_mcs)

    print(f"P_FAILURE ==> {p_failures}")
    # print(f"V_COEFF ==> {v_coeffs}")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.3, 3.3)
    if plot:
        az.plot_dist(
            d,
            ax=ax,
            quantiles=[0.25, 0.5, 0.75],
            plot_kwargs={"color": "firebrick", "label": n},
            fill_kwargs={"alpha": 0.4, "color": "palegreen"},
        )

        ax.hist(ppc.observed_data, label=f"Daño Observado {n}")
        ax.set_xlabel(n)
        ax.set_xlim(0, None)

        plt.savefig(RESULTS_FOLDER / f"partial_damage_{n}_{dmg_model}.png", dpi=600)
        plt.close()

    return {"p_failure": p_failures,
            "d_mean":d.values.mean(), "d_std": d.values.std()}


def main(year_init=0, year_end=N_YEARS, plot=False):
    p_failures_total = {"miner": [], "Aeran": []}
    p_failure_path = RESULTS_FOLDER / "P_FAILURE_PLOT_ACCUMULATED.png"
    p_failure_arr = RESULTS_FOLDER / "PFAILURES.npz"
    d_means_total = {"miner": [], "Aeran": []}
    d_stds_total = {"miner": [], "Aeran": []}



    if not os.path.exists(p_failure_arr):
        for year_batch in window(range(year_init, year_end), 4):
            year_init, year_end = year_batch[0], year_batch[-1]
            print(f"YEAR_BATCH => {year_batch}")

            for dmg_model in ["miner", "Aeran"]:
                strt = perf_counter()
                damage_model = build_damage_model(year_init, year_end)
                trace = sample_model(
                    damage_model, year_init=year_init, year_end=year_end, draws=1000
                )
                end = perf_counter()
                print_perf("SAMPLING", strt, end)

                strt = perf_counter()
                print("SAMPLING POSTERIOR ==> ")
                ppc, partial_names = posterior_sample(damage_model, trace, year_init, year_end)
                end = perf_counter()
                print_perf("PPC", strt, end)

                # fig, ax = plt.subplots(len(partial_names))
                # fig.set_size_inches(3.1, 6.3)
                # plt.subplots_adjust(wspace=0.05175)
                strt = perf_counter()
                args = [(ppc, n, plot, dmg_model) for n in partial_names]
                results = list(map(lambda a: post_process(*a), args))
                p_failures = [r["p_failure"] for r in results]
                d_means = [r["d_mean"] for r in results]
                d_stds = [r["d_std"] for r in results]
                p_failures_total[dmg_model].extend(p_failures)
                d_means_total[dmg_model].extend(d_means)
                d_stds_total[dmg_model].extend(d_stds)
                end = perf_counter()
                print_perf("POST_PROCESS", strt, end)


        # monkey patch it
        p_failures_total = {k:np.array(v) for k,v in p_failures_total.items()}
        np.savez_compressed(p_failure_arr, p_failures_total)
    else:
        p_failures_total = np.load(p_failure_arr, allow_pickle=True)['arr_0']

    print(p_failures_total)

    x = np.arange(1980, 1980+len(p_failures_total["miner"]))
    fig, (tax, bax) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(3.3, 6.3)
    # Just lazy
    aux = {"miner":"Miner", "Aeran":"Aeran"}
    colors = {"miner":"salmon", "Aeran":"lightgreen"}

    for dmg_model in ["miner", "Aeran"]:

        tax.plot(x,p_failures_total[dmg_model], label=f"{aux[dmg_model]}")
        tax.scatter(x, p_failures_total[dmg_model])
        tax.set_xlabel("Año")
        tax.set_ylabel(r"$\mathrm{P}_{falla}$")
        bax.plot(x, d_means_total[dmg_model], label=f"{aux[dmg_model]}")
        bax.fill_between(x,
                         np.array(d_means_total[dmg_model]) + np.array(d_stds_total[dmg_model][0]),
                         np.array(d_means_total[dmg_model]) - np.array(d_stds_total[dmg_model][0]),
                         alpha=0.4, color=colors[dmg_model], label=fr"Region $1\sigma$")
        bax.set_xlabel("Año")
        bax.set_ylabel(r"$D$")

        bax.hlines(y=1, xmin=x[0], xmax=x[-1],  linestyle="dashed")
        bax.set_ylim(0, 1.3)

        tax.legend()
        bax.legend()
    plt.savefig(RESULTS_FOLDER / "PFAILURES_MEAN.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
