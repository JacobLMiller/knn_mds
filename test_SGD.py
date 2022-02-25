


def grad_for_xi(X,d,w,i):
    return alpha*stress_grad(X,d,w,i) + (1-alpha)*local_grad(X,d,w,i)

def stress_grad(X,d,w,i):
    grad = 0
    for j in range(len(X)):
        if i != j:
            mag = np.linalg.norm(X[i]-X[j])
            grad += ((mag - di)/2)
