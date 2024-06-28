from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

from autograd import jacobian
import autograd.numpy as np  # Import wrapped numpy from autograd

from func_timeout import func_timeout, FunctionTimedOut


def optimise_latent_at_distance(l1, d1, x0=None, options=None):
    
    def f(x):
        e1 = np.sqrt(np.sum(np.square(x - l1)))
        cost = np.square(np.abs(e1-d1))
        return cost
        
    if x0 is None:
        x0 = l1 + np.random.randn(l1.shape[0])
        
    res = minimize(f, x0, method='Newton-CG', jac=jacobian(f), options=options)

    return res.x, euclidean(l1, res.x), res


def optimise_latent_dataset_at_euclideans(reference_latent, desired_euclideans_from_target, epsilon=0.01):

    dataset_latents = []
    dataset_euclideans = []
    N = len(desired_euclideans_from_target)
    for i in range(N):

        import sys
        sys.stdout.write(f'\rIteration {i+1}/{N}')  # Print the current iteration
        sys.stdout.flush()
        
        while True:
            terminated = False
            try:
                proposed_latent, proposed_euclidean, res = func_timeout(0.1, optimise_latent_at_distance, (reference_latent, desired_euclideans_from_target[i]))
                terminated = True
            except FunctionTimedOut:
                # print("Optimization was terminated due to timeout.")
                pass
                
            if terminated:
                break

        dataset_latents.append(proposed_latent)
        dataset_euclideans.append(proposed_euclidean)

    return dataset_latents, dataset_euclideans