from aub_htp import alpha_stable
from scipy.stats import levy_stable

alpha = 1.2
beta = 0.3


ok = alpha_stable.rvs(alpha, beta, size = 10)

print(ok)