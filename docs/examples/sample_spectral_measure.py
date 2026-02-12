from aub_htp.random.spectral_measure_sampler import IsotropicSampler, EllipticSampler, MixedSampler
from aub_htp.random.alpha_stable_sampler import sample_alpha_stable_vector


isotropic_sampler = IsotropicSampler(2, 1)
print(isotropic_sampler.sample())

sigma = [[4, 2],
         [2, 3]]
elliptic_sampler = EllipticSampler(2, sigma, total_mass=1)
print(elliptic_sampler.sample())

mixed_sampler = MixedSampler([isotropic_sampler, elliptic_sampler], [0.1, 0.9])
print(mixed_sampler.sample())

x = sample_alpha_stable_vector(100, 1.2, mixed_sampler, )
print(x)