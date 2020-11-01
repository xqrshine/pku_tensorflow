import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack((a, b)))

mgrid = np.mgrid[1:10:1, 0:1:0.1]
print(mgrid)
print(mgrid.shape)

print(mgrid.ravel())