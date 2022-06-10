#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

n = 100000

# generate some random multimodal histogram data
samples = np.concatenate([np.random.normal(np.random.randint(-8, 8), size=n)*np.random.uniform(.4, 2) for i in range(4)])
h,e = np.histogram(samples, bins=100, density=True)
x = np.linspace(e.min(), e.max())

# plot the histogram
plt.figure(figsize=(8,6))
plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')

# plot the real KDE
kde = sts.gaussian_kde(samples)
plt.plot(x, kde.pdf(x), c='C1', lw=8, label='KDE')
#%%
# resample the histogram and find the KDE.
resamples = np.random.choice((e[:-1] + e[1:])/2, size=n*5, p=h/h.sum())
rkde = sts.gaussian_kde(resamples)

# plot the KDE
plt.plot(x, rkde.pdf(x), '--', c='C3', lw=4, label='resampled KDE')
plt.title('n = %d' % n)
plt.legend()
plt.show()
# %%
