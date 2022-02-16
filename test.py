import numpy as np

v = np.array([0.1,0])
u = np.array([0,0.1])
t = 1
print(v,u)


pq = v-u
mag = np.linalg.norm(pq)

r = 1/(mag*pq)

m = -(t)*r

print('Magnitude:', mag)
print("Repulsion:", np.linalg.norm(m))

A = [np.ones((3,3)) for i in range(5)]
print(type(sum(A)))
