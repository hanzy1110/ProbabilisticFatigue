#!/home/blueman69/miniforge3/envs/fem_gp/bin/python

import os
import pathlib
import re
import pprint
from typing import Dict
import numpy as np
import pytensor
import arviz as az
import jax.numpy as jnp
from jax import Array, devices, local_devices
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])
import fire
from src.combinedModel import DamageCalculation
from src.freqModel import total_cycles_per_year


# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     jax.device_count()
# )
BASE_PATH = pathlib.Path(__file__).parent
DATA_DIR = BASE_PATH / "data"
RESULTS_FOLDER = BASE_PATH / "RESULTS"
PLOT_DIR = BASE_PATH / "plots"

LOAD_PATH = RESULTS_FOLDER / "LOADS"
CYCLING_HOURS = 500
N_YEARS = 15
N_DISTINCT_LOADS = 32

RANDOM_SEED = 9927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

printer = pprint.PrettyPrinter(5, compact=True)


def getPFailure(damages: Array):
    return len(damages[jnp.isclose(damages, 1)]) / len(damages)


def getVarCoeff(p_failures, N_mcs):
    return np.sqrt((1 - p_failures) / (N_mcs * p_failures))


def plot_p_failure(values: Dict[int, Dict[str, np.float32]]):
    p_failures = []
    var_coeffs = []
    for year, vals in values.items():
        p_failures.append(np.array(vals["pFailures"]).mean())
        var_coeffs.append(np.array(vals["varCoeff"]).mean())

    fig, (tax, bax) = plt.subplots(2, 1)
    fig.set_size_inches(3.3, 6.3)
    tax.plot(p_failures)
    tax.set_xlabel("Year")
    tax.set_ylabel(r"$\mathrm{P}_{failure}$")
    bax.plot(var_coeffs)
    bax.set_xlabel("Year")
    bax.set_ylabel(r"$\delta_{\mathrm{P}_{failure}}$")
    plt.savefig(RESULTS_FOLDER / "p_failure_plot.png", dpi=600)
    plt.close()


def delete_files_by_regex(folder_path, regex_pattern):
    try:
        # Check if the provided folder path exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        # List all files in the folder
        files = os.listdir(folder_path)

        # Compile the regex pattern
        pattern = re.compile(regex_pattern)

        # Iterate through files and delete those matching the regex pattern
        for file_name in files:
            if pattern.match(file_name):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("Deletion completed.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Re sample posterior to plot properly
# @profile
def main(T: int, ndraws_wohler: int, delete_files: bool = False, debug: bool = False):
    # print(T)
    # T = int(T[0])
    file_pattern = r"tot_damages.*.npz"

    if delete_files:
        delete_files_by_regex(LOAD_PATH, file_pattern)

    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    print(f"Running on: {devices()}")
    scaleFactors = np.arange(10, 10000, 100)

    print("-x" * 30)
    print(f"T:{T}")
    props = {
        "Eal": 69e3,
        "Ecore": 207e3,
        "rw": 2.5,
        "rc": 1,
        "layers": [26, 7],
        # 'T':30641*0.15
        "T": 97.4 * 1000 * T / 100,
    }

    damageCal = DamageCalculation(
        wohlerPath=RESULTS_FOLDER / "WOHLER",
        loadObserved=DATA_DIR / "800369.csv",
        WohlerObserved=DATA_DIR / "SN_curve.mat",
        loadPath=RESULTS_FOLDER / "LOADS",
        cableProps=props,
        PFObserved=DATA_DIR / "pfData.csv",
        Tpercentage=T,
    )

    damageCal.restoreTraces(ndraws_wohler=ndraws_wohler)
    print("max vals WohlerC-->")
    print("logN-->", damageCal.WohlerC.NMax)
    print("SMax-->", damageCal.WohlerC.SMax)
    print("max vals Loads-->", damageCal.LoadM.maxAmp)
    # damageCal.sample_model('wohler', 2000)
    # damageCal.sample_model('loads', 2000)

    cycles_per_year = total_cycles_per_year(
        CYCLING_HOURS, N_YEARS, DATA_DIR / "800369.csv"
    )

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.3, 3.3)
    ax.plot(cycles_per_year)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cycles / Year")
    ax.set_yscale("log")
    plt.savefig(RESULTS_FOLDER / "cycles_plot.png", dpi=600)
    plt.close()

    print(f"CYCLES PER YEAR {cycles_per_year[0]}")

    damageCal.restoreLoadSamples(ndraws=cycles_per_year[0])
    damageCal.restoreFatigueLifeSamples(maxLoads=N_DISTINCT_LOADS)
    damageCal.plotFatigueLifeSamples()

    print("Starting Damage Calculation...")
    vals = {"pFailures": [], "varCoeff": []}

    # if debug:
    #     damages_aeran = damageCal.calculate_damage_debug(cycles_per_year[0])
    #     print(damages_aeran)

    # else:

    vals = {i: {"pFailures": [], "varCoeff": []} for i in range(len(cycles_per_year))}
    for year, ncycles in enumerate(cycles_per_year):
        # cycles = jnp.array(damageCal.cycles, dtype=jnp.float32) * cycles_per_year

        if not os.path.exists(LOAD_PATH / f"tot_damages_year{year}_batch_0.npz"):
            nbatches = damageCal.calculate_damage(
                cycles_per_year=ncycles, year=year, nloads=N_DISTINCT_LOADS
            )
        else:
            nbatches = 100

        tot_damages = None
        for i in range(nbatches):
            print(f"BATCH NUMBER : {i}")

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

                vals[year]["pFailures"].append(getPFailure(damages_aeran))
                N_mcs = len(damages_aeran)
                if np.isclose(vals[year]["pFailures"][-1], 0):
                    vals[year]["varCoeff"].append(0)
                else:
                    vals[year]["varCoeff"].append(
                        getVarCoeff(vals[year]["pFailures"][-1], N_mcs)
                    )
            except Exception as e:
                print(e)
                raise e

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(6.3, 6.3)

        p_failures = vals[year]["pFailures"]
        var_coeff = vals[year]["varCoeff"]
        ax[0].hist(p_failures, label="Probability of failure")
        ax[1].hist(tot_damages, density=True, bins=100, label="Aeran Damages")
        # print(f"DAMAGES MEAN => {tot_damages.mean()}")
        # print(f"DAMAGES STD => {tot_damages.std()}")
        # ax[0].set_xlabel("Probability of Failure")

        ax[0].legend()
        ax[1].legend()
        plt.savefig(RESULTS_FOLDER / f"p_failure_year_{year}.png", dpi=600)
        plt.close()

    plot_p_failure(vals)


if __name__ == "__main__":
    # T = 20
    # ax.set_yscale('log')
    fire.Fire(main)
