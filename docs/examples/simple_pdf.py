import aub_htp as ht

x = [1,2,3,4]

y = ht.alpha_stable.with_parametrization("S0").pdf(x, alpha = 1, beta = 1, loc = 1, scale = 3)




print(y)

