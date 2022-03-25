def example():
    import numpy as np
    import itertools
    import time

    times = []

    n = 100

    indices = list(itertools.combinations(range(n), 2))

    d = np.ones((n,n))
    X = np.random.uniform(0,1,2*n)

    step = 0.1

    shuffle = np.random.shuffle
    for epoch in range(100):

        for i,j in indices:
            #old = np.linalg.norm(X[i]-X[j])


            start = time.perf_counter()
            pq = X[i]-X[j]
            end = time.perf_counter()
            times.append(end-start)
            mag = (pq[0]*pq[0] + pq[1]*pq[1]) ** 0.5

            mag_grad = pq/mag

            #w = 1/(dij**2)
            mu = step
            if mu >= 1: mu = 1

            r = (mu*(mag-d[i][j]))/(2*mag)
            stress = r*pq

            mu1 = step if step < 1 else 1
            repulsion = -((step/mag) * mag_grad)
            # if mag > 3*diam:
            #     repulsion *= 1
            #
            l_sum = 1+0.01
            m = (1/l_sum)*stress + (0.5/l_sum)*repulsion

            X[i] -= m
            X[j] += m


        shuffle(indices)

    print(sum(times)/len(times))

import numpy as np

mn = np.triu_indices(50)
for i in mn:
    print(i)

X = np.random.uniform(0,1,(100,2))
