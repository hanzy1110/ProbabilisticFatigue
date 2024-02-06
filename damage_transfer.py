import os
import pathlib
import fire
import pymc as pm
import nutpie
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import collections, itertools
import scienceplots

from scipy import stats
from pymc.distributions import Interpolated

plt.style.use(["science", "ieee"])

BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"
LOAD_PATH = RESULTS_FOLDER / "LOADS"
MAX_SAMPLES = 25000

N_YEARS = 50
N_BATCHES = 100


n_min = 1
n_max = 100
n_batches = 5


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
    x = np.concatenate([[0], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)


def get_prev_damage(year_init, year_end):
    prev_year_end = year_init + 1
    prev_year_init = prev_year_end - (year_end - year_init)
    ppc_filename = (
        RESULTS_FOLDER / f"damage_posterior_{prev_year_init}_{prev_year_end}.nc"
    )

    print(f"PPC FILENAME ==> {ppc_filename}")

    d_name = f"damage_{year_init}-{year_init-1}"
    prev_ppc = az.from_netcdf(ppc_filename)

    prev_damage_samples = prev_ppc.posterior_predictive[d_name].mean(
        dim=("chain", "draw")
    )

    return from_posterior(d_name, prev_damage_samples)


def get_tot_damages(year) -> np.ndarray:
    tot_damages = None
    for i in range(N_BATCHES):
        # print(f"BATCH NUMBER : {i}")

        try:
            with open(
                LOAD_PATH / f"tot_damages_year{year}_batch_{i}.npz", "rb"
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
        print("NAN FRAC ==>")
        print(f"NAN LEN {len(tot_damages[np.isnan(tot_damages)])}")
        print(len(tot_damages[np.isnan(tot_damages)]) / len(tot_damages))
        print(f"TOTAL LEN {len(tot_damages)}")
        return tot_damages[~np.isnan(tot_damages)][:MAX_SAMPLES]

    return tot_damages


def build_damage_model(year_init, year_end):
    year_range = range(year_init, year_end)
    damages = [get_tot_damages(i) for i in year_range]
    tot_damages = np.array([d for d in damages if d is not None], dtype=np.float32)

    # with pm.Model(coords=coords) as damage_model:
    with pm.Model() as damage_model:
        for j, i in enumerate(year_range):
            alpha = pm.Gamma(f"alpha_{i}", alpha=1, beta=1)
            beta = pm.Gamma(f"beta_{i}", alpha=1, beta=1)
            damages = pm.Gamma(
                f"damage_{i}",
                alpha=alpha,
                beta=beta,
                observed=tot_damages[j, :],
            )

    return damage_model


def sample_model(damage_model, draws, year_init, year_end):
    filename = RESULTS_FOLDER / f"DAMAGE_MODEL_TRACE_{year_init}_{year_end}.nc"

    if not os.path.exists(filename):
        compiled_model = nutpie.compile_pymc_model(damage_model)
        trace = nutpie.sample(compiled_model, draws=draws, tune=1000, chains=4)
        az.to_netcdf(
            data=trace,
            filename=filename,
        )
    else:
        print("LOADING TRACE")
        trace = az.from_netcdf(filename)

    return trace


def posterior_sample(damage_model, trace, year_init, year_end):
    partial_year_range = range(year_init + 1, year_end)
    year_range = range(year_init, year_end)
    partial_names = [f"damage_{i}-{i-1}" for i in partial_year_range]
    names = [f"damage_{i}" for i in year_range]

    print(names)
    print(partial_names)

    ppc_filename = RESULTS_FOLDER / f"damage_posterior_{year_init}_{year_end}.nc"

    if not os.path.exists(ppc_filename):
        with damage_model:
            for i, n in enumerate(names[1:]):
                if i == 0:
                    if year_init != 0:
                        damage_prev = get_prev_damage(year_init, year_end)
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


def main(year_init=0, year_end=N_YEARS):
    for year_batch in window(range(year_init, year_end), 4):
        year_init, year_end = year_batch[0], year_batch[-1]

        print(year_init, year_end)

        print(year_batch)

        damage_model = build_damage_model(year_init, year_end)
        trace = sample_model(
            damage_model, year_init=year_init, year_end=year_end, draws=1000
        )
        ppc, partial_names = posterior_sample(damage_model, trace, year_init, year_end)

        fig, ax = plt.subplots(len(partial_names))
        fig.set_size_inches(3.1, 6.3)
        plt.subplots_adjust(wspace=0.05175)
        for i, n in enumerate(partial_names):
            d = ppc.posterior_predictive[n]
            az.plot_dist(
                d,
                ax=ax[i],
                quantiles=[0.25, 0.5, 0.75],
                plot_kwargs={"color": "darkcoral", "label": n},
                fill_kwargs={"alpha": 0.3, "color": "firebrick"},
            )
            ax[i].set_xlabel(n)
            ax[i].set_xlim(0, 3)

        plt.savefig(
            RESULTS_FOLDER / f"partial_damage_{year_init}_{year_end}.png", dpi=600
        )
        plt.close()


if __name__ == "__main__":
    fire.Fire(main)
