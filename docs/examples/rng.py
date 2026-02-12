from aub_htp.random.rng import *


def pdf_cauchy_1d(x, gamma):
    return 1.0 / (math.pi * gamma * (1.0 + (x / gamma) ** 2))


def pdf_independent_cauchy_2d(x, gamma1, gamma2):
    x = np.atleast_2d(x)
    return pdf_cauchy_1d(x[:, 0], gamma1) * pdf_cauchy_1d(x[:, 1], gamma2)


def pdf_cauchy_nd(x, d, gamma):
    x = np.atleast_2d(x)
    r2 = np.sum(x ** 2, axis=1)
    const = math.gamma((d + 1) / 2.0) / (
            (math.pi ** ((d + 1) / 2.0)) * (gamma ** d)
    )
    denom = (1 + r2 / (gamma ** 2)) ** ((d + 1) / 2.0)
    return const / denom


def closed_entropy_isotropic_cauchy(d, gamma):
    term1 = (
            (d + 1) / 2.0
            * (math.log(4 * math.pi) + s.EulerGamma.evalf() + psi((d + 1) / 2.0))
    )
    term2 = -gammaln((d + 1) / 2.0)
    return term1 + term2 + d * math.log(gamma)


Sigma1 = np.array([[2, 0.8],
                   [0.8, 1.5]])  # horizontal major axis

Sigma2 = np.array([[2, -1],
                   [-1, 2]])  # vertical major axis

alpha = 1
N = 500
M = 500000
d = 2
masses = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
weight = np.array([0.25, 0.25, 0.25, 0.25])
# X = Sample_Multivariate_LePage(alpha=alpha,N=N,gamma=2,M=M,d=2,spectral_measure="isotropic",delta=np.array([1000 for i in range(d)]))
X = Sample_Multivariate_LePage(alpha, N, d, M, spectral_measure="isotropic", delta=np.array([1000 for i in range(d)]),
                               sigma=Sigma2, )


# X = Sample_Multivariate_LePage(alpha,N,d,M,spectral_measure="atomic",masses=masses,weights_of_masses=weight,delta=np.array([1000 for i in range(d)]))

def sampler1(k):
    return Isotropic_Spectral_Measure(d, k)


def sampler2(k):
    return Sample_LePage_One_Point_Discrete(k, masses=masses, weights_of_masses=weight)


def sampler3(k):
    return Subguassian_Spectral_Measure(k, d, Sigma1)


def sampler4(k):
    return Subguassian_Spectral_Measure(k, d, Sigma2)

SM1 = SpectralMeasure(d, sampler1, 1)
SM2 = SpectralMeasure(d, sampler2, 1)
SM3 = SpectralMeasure(d, sampler3, Estimate_Spectral_Measure_Subguassian(alpha, d, 200000, Sigma1))
SM4 = SpectralMeasure(d, sampler4, Estimate_Spectral_Measure_Subguassian(alpha, d, 200000, Sigma2))
spec_measures = [SM1, SM4]
# X=Sample_Multivariate_LePage(alpha,N,d,M,"mixed",delta=np.array([1000 for i in range(d)]),spectral_measures=spec_measures,weights_of_spectral_measures=np.array([0.5,0.5]))
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=0.1, alpha=0.3, color='blue')
plt.grid(True)
plt.axis("equal")
plt.xlim((-400 + 1000, 400 + 1000))
plt.ylim((-400 + 1000, 400 + 1000))
plt.title("Isotropic 2D Cauchy (LePage)")
plt.show()
