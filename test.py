import collections, itertools
import numpy as np

n_min = 1
n_max = 100
n_batches = 5

def window(it, winsize, step=2):
    """Sliding window iterator."""
    it=iter(it)  # Ensure we have an iterator
    l=collections.deque(itertools.islice(it, winsize))
    while 1:  # Continue till StopIteration gets raised.
        try:
            yield tuple(l)
            for i in range(step):
                l.append(next(it))
                l.popleft()
        except StopIteration as e:
            return

print(list(window(range(n_min, n_max),3)))
