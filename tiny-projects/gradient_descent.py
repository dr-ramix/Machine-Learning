import numpy as np

def gradient_descent_nd(gradient_func, start, learning_rate=0.1, tolerance=1e-6, max_iters=1000):
    x = np.array(start, dtype=float)
    for i in range(max_iters):
        grad = np.array(gradient_func(x))
        new_x = x - learning_rate * grad

        if np.linalg.norm(new_x - x) < tolerance:
            print(f"Converged in {i} iterations.")
            break

        x = new_x

    return x



def grad_f_nd(xy):
    x, y = xy
    return [2 * x, 2 * y]

minimum = gradient_descent_nd(grad_f_nd, start=[3.0, 4.0])
print(f"Minimum found at x = {minimum}")
