import numpy as np
import scipy.io
from paper_style import paper_figure

## Figure 5 of the paper with the graded parsimonious RadauIIA-3, from the data of
## convplots_RadauIIA3_graded.py (run it first with: python3 convplots_RadauIIA3_graded.py 0.0004).
## Branch 2 is obtained from branch 0 by the symmetry s_2(l) = conj(s_0(1-l)) and branch 1 is
## self-symmetric; Laplace evaluations are counted on one side of the real axis only
## (L = m*N for classic CQ). Reference lines: L^{-p} for classic CQ and L^{-p*2m/(2m-1)} for the
## parsimonious method, with p = 5/4 (RadauIIA-3). N = 64, ..., 32768.
ct = 0.0004
d = scipy.io.loadmat("../scalar/data/err_RadauIIA3_graded_HS_ct"+str(ct)+".mat")
v = lambda k: d[k].ravel()
N = np.round(5.0/v("taus")).astype(int)
I = N >= 64
paper_figure("RadauIIA-3",v("Ns")[I],v("err")[I],v("counters_graded")[I],v("errg")[I],
             5.0/4,r"$\mathcal{O}(L^{-5/4})$",N[I],v("devg")[I],"paper_errs_RadauIIA3.pdf",ref_shift=1.25,xlim=(1e2,1.5e5),
             pars_ref=(5.0/4*6/5,r"$\mathcal{O}(L^{-3/2})$",0.6))
