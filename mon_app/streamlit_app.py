import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import pyximport, numpy
pyximport.install(language_level=3, setup_args={"include_dirs": numpy.get_include()})
import sim

st.set_page_config(page_title="Tournoi Cython", layout="centered")
st.title("Simulation Tournoi (Cython)")

levels = {
    1: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 0.5},
    2: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 0.6},
    3: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 0.75},
    4: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 1},
    5: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 1.5},
    6: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 2.5},
    7: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 5},
    8: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 10},
    9: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 20},
    10: {"multiplicateurs": {0.4: 0.6, 1: 0.223, 2: 0.126, 3: 0.045, 10: 0.005, 100: 0.001}, "buy_in_required": 50},
    11: {"multiplicateurs": {0.5: 0.6, 1: 0.23, 2: 0.11, 3: 0.05, 10: 0.01}, "buy_in_required": 100},
    12: {"multiplicateurs": {0.6: 0.6, 1: 0.22, 2: 0.12, 3: 0.06}, "buy_in_required": 200},
    13: {"multiplicateurs": {0.7: 0.53340, 1: 0.30658, 2: 0.16002}, "buy_in_required": 500},
    14: {"multiplicateurs": {0.8: 0.46425, 1: 0.35005, 1.5: 0.18570}, "buy_in_required": 1000},
}

if "levels_loaded" not in st.session_state:
    sim.load_levels(levels)
    st.session_state["levels_loaded"] = True

with st.sidebar:
    space_ko = st.checkbox("space_ko", value=True)
    n_joueurs = st.slider("n_joueurs", 100, 3000, 1000, 100)
    iterations = st.slider("iterations", 10, 2000, 200, 10)
    stack_initial = float(st.number_input("stack_initial", value=20000.0, step=1000.0))
    prime_initiale = float(st.number_input("prime_initiale", value=5.0, step=0.5))
    seed = st.number_input("seed (0 = aléatoire)", value=0, step=1)

if st.button("Lancer"):
    t0 = time.perf_counter()
    totals = np.empty(iterations, dtype=float)
    for i in range(iterations):
        _, total = sim.simuler_tournoi_cython(
            n_joueurs=n_joueurs,
            space_ko=space_ko,
            stack_initial=stack_initial,
            prime_initiale=prime_initiale,
            seed=int(seed),
        )
        totals[i] = total
    dt = time.perf_counter() - t0

    mean_total = float(totals.mean())
    st.success(f"{iterations} itérations en {dt:.2f}s  •  Moyenne = {mean_total:.2f}")

    fig = plt.figure()
    plt.hist(totals, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(mean_total, linestyle="--", linewidth=2, label=f"Moyenne = {mean_total:.2f}")
    plt.title("Distribution de total_gains")
    plt.xlabel("total_gains")
    plt.ylabel("Fréquence")
    plt.legend()
    st.pyplot(fig)
