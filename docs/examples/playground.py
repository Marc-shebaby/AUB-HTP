from aub_htp.random import BaseSpectralMeasureSampler, IsotropicSampler, EllipticSampler, sample_alpha_stable_vector


isotropic = IsotropicSampler(2, 1.2)
elliptic = EllipticSampler(2, [[10, 2], [2, 50]])

samples = sample_alpha_stable_vector(1.2, isotropic, 10)

print(samples)