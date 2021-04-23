#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np

#%%
a = np.array([[0, -1],[1, 0]])
t = np.array([[3,4],[4,-3]])
a_inv = np.linalg.inv(a)
x = np.array([[-2],[4]])
y = np.array([[2],[-1]])
a_inv.dot(x)
a.dot(x)
a_inv.dot(t).dot(a)

#%%
np.linalg.svd(np.array([[6,2],[-7,6]]))

#%%
import heapq

# a = np.asarray([1, 2, 3, 4])
# b = np.asarray([[1, 2, 3, 4],[5, 10, 3, 4]])
# candidates = heapq.nlargest(2, b, key=lambda x: x[1])

# arr2 = np.vectorize(lambda coord: dist(1, 1, coord[0], coord[1]))(arr)
#  arr3 = np.argsort(arr2)
#  arr = np.array(arr)[arr3]



#%%

a = np.array([[1, 2.5, 3, 4],[5, 2.5, 3, 4],[50, 109, 93, 41]])
point = np.array([1, 2, 3, 4])
d = ((a-point)**2).sum(axis=1)  # compute distances
ndx = d.argsort() # indirect sort 

a[ndx[:2]]
d[ndx[:2]]
# print 10 nearest points to the chosen one
# import pprint
# pprint.pprint(zip(a[ndx[:2]], d[ndx[:2]]))
