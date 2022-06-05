import numpy as np

def hist_sample(hist, n):
    """Genertae histogram sample
    Args:
        hist (array): hist[0]: frecuencies/probs of X values, hist[1]: X values
        n ([type]): number of samples
    Returns:
        [list]: list with samples
    """
    return np.random.choice(hist[1], size=n, p=hist[0]/sum(hist[0]))

# generate sample from histogram
sample_2 = hist_sample(hist, sum(hist[0])*100)
