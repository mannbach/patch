from netin.models import PATCHModel, CompoundLFM

from patch.elfi import compute_m

def test_compute_m():
    for n in [10, 100, 1000, 5000]:
        for m in [2, 4, 8]:
            model = PATCHModel(
                f_m=.2,
                N=n,
                m=m,
                tau=0.,
                lfm_global=CompoundLFM.UNIFORM,
                lfm_tc=CompoundLFM.UNIFORM,
            )
            graph = model.simulate()

            m_comp = compute_m(graph_empirical=graph, n_nodes_sim=n)
            assert m_comp == m, (
                f"Expected m={m}, but got {m_comp} for n={n}, m={m}."
            )
