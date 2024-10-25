# %%
import elfi
import matplotlib.pyplot as plt
import numpy as np
from netin.models import PATCHModel, CompoundLFM
from patch.statistics import compute_gini, compute_ei, compute_mann_whitney
from patch.constants import F, M

N_SAMPLES = 1000

N_SIM=1000
N_TRUE = 10000
LFM_TC = CompoundLFM.UNIFORM
LFM_GLOBAL = CompoundLFM.PAH

H_TRUE = 0.25
TAU_TRUE = 0.75

def simul(h: float, tau: float, N: int = None, random_state=None):
    model = PATCHModel(
        N=N_SIM if N is None else N,
        f_m=F, m=M, tau=float(tau), h_M=float(h), h_m=float(h), random_state=random_state, lfm_global=LFM_GLOBAL, lfm_tc=LFM_TC)
    g = model.simulate()
    return g

def f_gini(g):
    return compute_gini(g.degrees())
def f_ei(g):
    return (compute_ei(g) + 1) / 2
def f_mw(g):
    return compute_mann_whitney(g)

def main():
    # elfi.set_client('multiprocessing')

    rng = np.random.RandomState(0)

    h_pr = elfi.Prior('uniform', 0, 1)
    tau_pr = elfi.Prior('uniform', 0, 1)

    g_true = simul(h=H_TRUE, tau=TAU_TRUE, N=N_TRUE, random_state=rng)
    gini_true = f_gini(g_true)
    ei_true = f_ei(g_true)
    mw_true = f_mw(g_true)

    sim = elfi.Simulator(
        elfi.tools.vectorize(simul, dtype=False),
        h_pr, tau_pr,
        observed=g_true)

    s_gini = elfi.Summary(elfi.tools.vectorize(f_gini), sim)
    s_ei = elfi.Summary(elfi.tools.vectorize(f_ei), sim)
    s_mw = elfi.Summary(elfi.tools.vectorize(f_mw), sim)
    d = elfi.Distance('euclidean', s_gini, s_ei, s_mw)

    rej = elfi.Rejection(d)

    sample = rej.sample(n_samples=N_SAMPLES)

    print(sample.summary())

    plt.scatter(
        sample.samples['h_pr'], sample.samples['tau_pr'], c=sample.outputs['d'])

    plt.axhline(TAU_TRUE, color="black", linestyle="--")
    plt.axvline(H_TRUE, color="black", linestyle="--")

    plt.title(
        f"$ei_{{sc}}={ei_true:.2f},\\ gini={gini_true:.2f},\\ mw={mw_true:.2f}$")
    plt.xlabel('$h$')
    plt.ylabel('$\\tau$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('scatter.pdf')

if __name__ == "__main__":
    main()
