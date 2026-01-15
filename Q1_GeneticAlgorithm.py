import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_lab_test_q1() -> GAProblem:
    dim = 80  
    target = 40  
    max_fit = 80.0  

    def fitness(x: np.ndarray) -> float:
        ones = float(np.sum(x))
        dist = abs(ones - target)
        return max_fit - dist  # The coding changes: Fitness formula

    return GAProblem(
        name="Lab Test Q1 (Target 40 Ones)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )

def make_onemax(dim: int) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        return float(np.sum(x))  # maximize number of ones

    return GAProblem(
        name=f"OneMax ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


def make_sphere(dim: int, lo: float, hi: float) -> GAProblem:
    # We will maximize negative sphere to convert to a maximization problem
    def fitness(x: np.ndarray) -> float:
        return -float(np.sum(np.square(x)))

    return GAProblem(
        name=f"Sphere {dim}D (maximize -||x||^2)",
        chromosome_type="real",
        dim=dim,
        bounds=(lo, hi),
        fitness_fn=fitness,
    )


def make_rastrigin(dim: int, lo: float, hi: float) -> GAProblem:
    def rastrigin(x: np.ndarray) -> float:
        A = 10.0
        return float(A * x.size + np.sum(x * x - A * np.cos(2 * np.pi * x)))

    def fitness(x: np.ndarray) -> float:
        return -rastrigin(x)  # maximize negative cost

    return GAProblem(
        name=f"Rastrigin {dim}D (maximize -f)",
        chromosome_type="real",
        dim=dim,
        bounds=(lo, hi),
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        assert problem.bounds is not None
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(a.shape) < p
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1.copy(), c2.copy()


def arithmetic_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    alpha = rng.random(a.shape)
    c1 = alpha * a + (1 - alpha) * b
    c2 = alpha * b + (1 - alpha) * a
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def gaussian_mutation(x: np.ndarray, mut_rate: float, sigma: float, rng: np.random.Generator, bounds: Tuple[float, float]) -> np.ndarray:
    y = x.copy()
    mask = rng.random(x.shape) < mut_rate
    noise = rng.normal(0.0, sigma, size=x.shape)
    y[mask] += noise[mask]
    lo, hi = bounds
    np.clip(y, lo, hi, out=y)
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    real_sigma: float,
    seed: int | None,
    stream_live: bool = True,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # Live UI containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best: List[float] = []
    history_avg: List[float] = []
    history_worst: List[float] = []

    for gen in range(generations):
        # Logging
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Live updates
        if stream_live:
            df = pd.DataFrame(
                {
                    "Best": history_best,
                    "Average": history_avg,
                    "Worst": history_worst,
                }
            )
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.6f}**"
            )

        # Elitism: keep top E
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))
        elites_fit = fit[elite_idx].copy() if E > 0 else np.empty((0,))

        # Create next generation
        next_pop: List[np.ndarray] = []
        while len(next_pop) < pop_size - E:
            # Select parents
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                if problem.chromosome_type == "bit":
                    c1, c2 = one_point_crossover(p1, p2, rng)
                else:
                    c1, c2 = arithmetic_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            if problem.chromosome_type == "bit":
                c1 = bit_mutation(c1, mutation_rate, rng)
                c2 = bit_mutation(c2, mutation_rate, rng)
            else:
                assert problem.bounds is not None
                c1 = gaussian_mutation(c1, mutation_rate, real_sigma, rng, problem.bounds)
                c2 = gaussian_mutation(c2, mutation_rate, real_sigma, rng, problem.bounds)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Insert elites and finalize
        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    # Final metrics and best solution
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (Lab Test Q1)")

with st.sidebar:
    st.header("Problem")
    # The coding changes: Added "Lab Test Q1" to options
    problem_type = st.selectbox("Type", ["Lab Test Q1 (Target 40 Ones)", "OneMax (bits)", "Sphere (real)", "Rastrigin (real)"])

    if problem_type == "Lab Test Q1 (Target 40 Ones)":
        # The coding changes: Initialize specific Lab Test problem
        st.info("Fixed Params: Length=80, Target=40 Ones")
        problem = make_lab_test_q1()
    elif problem_type == "OneMax (bits)":
        dim = st.number_input("Chromosome length (bits)", min_value=8, max_value=4096, value=64, step=8)
        problem = make_onemax(int(dim))
    else:
        dim = st.number_input("Dimension", min_value=2, max_value=256, value=10, step=1)
        lo = st.number_input("Lower bound", value=-5.12)
        hi = st.number_input("Upper bound", value=5.12)
        if problem_type == "Sphere (real)":
            problem = make_sphere(int(dim), float(lo), float(hi))
        else:
            problem = make_rastrigin(int(dim), float(lo), float(hi))

    st.header("GA Parameters")
    # The coding changes: Default Population = 300
    pop_size = st.number_input("Population size", min_value=10, max_value=5000, value=300, step=10)
    # The coding changes: Default Generations = 50
    generations = st.number_input("Generations", min_value=1, max_value=10000, value=50, step=10)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.01, 0.005)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 100, 2)
    real_sigma = st.number_input("Real-valued mutation sigma", min_value=1e-6, value=0.1, format="%.6f")
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=2**32 - 1, value=42)
    live = st.checkbox("Live chart while running", value=True)

left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        result = run_ga(
            problem=problem,
            pop_size=int(pop_size),
            generations=int(generations),
            crossover_rate=float(crossover_rate),
            mutation_rate=float(mutation_rate),
            tournament_k=int(tournament_k),
            elitism=int(elitism),
            real_sigma=float(real_sigma),
            seed=int(seed),
            stream_live=bool(live),
        )

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Solution")
        st.write(f"Best fitness: {result['best_fitness']:.6f}")

        if problem.chromosome_type == "bit":
            bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
            st.code(bitstring, language="text")
            
            # The coding changes: Display interpretation for Lab Test Q1
            ones_count = int(np.sum(result['best']))
            st.write(f"Number of ones: {ones_count} / {problem.dim}")
            
            if problem.name.startswith("Lab Test"):
                if ones_count == 40:
                    st.success("Target Achieved! (Ones = 40)")
                else:
                    st.warning(f"Target missed by {abs(40 - ones_count)}")
        else:
            vec = result["best"].astype(float)
            st.write("x* =", np.array2string(vec, precision=4, suppress_small=True))
            if problem.name.startswith("Sphere"):
                st.write(f"Objective value (min sphere): {-result['best_fitness']:.6f}")
            elif problem.name.startswith("Rastrigin"):
                st.write(f"Objective value (min rastrigin): {-result['best_fitness']:.6f}")

with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness")
    if st.button("Show final population table"):
        pop = st.session_state.get("_final_pop")
        fit = st.session_state.get("_final_fit")
        if pop is None or fit is None:
            st.info("Run GA first to view the final population.")
        else:
            nshow = min(20, pop.shape[0])
            df = pd.DataFrame(pop[:nshow])
            df["fitness"] = fit[:nshow]
            st.dataframe(df, use_container_width=True)

# Store final population on each run for optional display
if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None

def _store_final(pop, fit):
    st.session_state["_final_pop"] = pop
    st.session_state["_final_fit"] = fit