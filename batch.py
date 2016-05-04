__author__ = 'jeremyma'
import config
from system import prototype_montecarlo_system
import multiprocessing
from functools import partial

if __name__ == '__main__':
    pool = multiprocessing.Pool(None)
    map_function = partial(prototype_montecarlo_system.calculate_samples, num_iterations=100, num_gaussians=8)
    r = pool.map(map_function, [('f0001',), ('f0002',)])
    r.wait() # Wait on the results
    print "done"