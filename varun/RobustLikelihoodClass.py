

import numpy as np


class LikelihoodEvaluator():
    def __init__(self, Xpoints, numMixtures):
        assert(Xpoints.ndim==2)

        self.Xpoints = Xpoints
        self.numPoints, self.dim = Xpoints.shape
        self.numMixtures = numMixtures

    def loglikelihood(self, means, diagCovs, weights):
        raise NotImplementedError

    __call__ = loglikelihood


class scikitLL(LikelihoodEvaluator):
    """
    Fastest Single Core Version so far!
    """


    def __init__(self, Xpoints, numMixtures):
        print "Scikits Learn Implementation Chosen"
        LikelihoodEvaluator.__init__(self, Xpoints, numMixtures)
        from sklearn.mixture import GMM as GMMEval
        self.evaluator = GMMEval(n_components=numMixtures)
        self.Xpoints = Xpoints


    def __str__(self):
        return "SciKit's learn implementation Implementation"


    def loglikelihood(self, means, diagCovs, weights):
        self.evaluator.weights_ = weights
        self.evaluator.covars_ = diagCovs
        self.evaluator.means_ = means

        return np.sum(self.evaluator.score(self.Xpoints))


class SingleCoreLL(LikelihoodEvaluator):

    def __init__(self, Xpoints, numMixtures):
        print "Single Core Implementation Chosen"
        LikelihoodEvaluator.__init__(self, Xpoints, numMixtures)

    def __str__(self):
        return "Single Core Implementation"



    def loglikelihood(self, means, diagCovs, weights):
        numMixtures = self.numMixtures



        #update if need be

        assert(means.shape == (numMixtures, self.dim))
        assert(diagCovs.shape == (numMixtures, self.dim))
        assert(len(weights)== numMixtures)


        numMixtures = len(weights)

        ll = np.zeros(self.numPoints)

        constMulti = self.dim / 2.0 * np.log(2 * np.pi)

        CovDet = np.zeros(numMixtures)

        for i in xrange(numMixtures):
            CovDet[i] = 1.0 / np.sqrt(np.prod(diagCovs[i]))

        for i in xrange(self.numPoints):
            for mixes in xrange(numMixtures):
                multiVal = 1

                temp = np.dot((self.Xpoints[i] - means[mixes]) / diagCovs[mixes], (self. Xpoints[i] - means[mixes]))
                temp *= -0.5
                ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]

            ll[i] = np.log(ll[i]) - constMulti

        return np.sum(ll)



class GPULL(LikelihoodEvaluator):

    def __str__(self):
        return "GPU Implementation"

    def __init__(self, Xpoints, numMixtures):
        print "GPU Implementation Chosen"
        LikelihoodEvaluator.__init__(self, Xpoints, numMixtures)

        #Pycuda imports
        import pycuda.autoinit
        from pycuda import gpuarray
        from pycuda.compiler import SourceModule

        self.gpuarray = gpuarray

        with open("KernelV2.cu") as f:

            if self.numPoints >= 1024:
                mod = SourceModule(f.read().replace('512', '1024'))
                self.numThreads = 1024
            else:
                mod = SourceModule(f.read())
                self.numThreads = 512

        if self.numPoints > self.numThreads:
            self.numBlocks = self.numPoints / self.numThreads
            if self.numPoints % self.numThreads != 0: self.numBlocks += 1
        else:
            self.numBlocks = 1

        print "numBlocks: {}, numPoints: {}".format(self.numBlocks, self.numPoints)
        #Set the right number of threads and blocks given the datasize
        #Using a max of 1024 threads, fix correct blocksize


        self.likelihoodKernel = mod.get_function("likelihoodKernel")
        self.likelihoodKernel.prepare('PPPPiiiP')

        self.Xpoints = self.Xpoints.astype(np.float32)
        self.Xpoints = gpuarray.to_gpu_async(self.Xpoints)

        self.means_gpu = gpuarray.zeros(shape = (self.numMixtures, self.dim), dtype = np.float32)
        self.diagCovs_gpu = gpuarray.zeros(shape = (self.numMixtures, self.dim), dtype = np.float32)
        self.weights_gpu = gpuarray.zeros(shape = self.numMixtures, dtype = np.float32)

        self.llVal = gpuarray.zeros(shape = self.numBlocks,  dtype=np.float32)

        #Allocate Memory for all our computations


    def loglikelihood(self, means, diagCovs, weights):


        assert(means.shape == (self.numMixtures, self.dim))
        assert(diagCovs.shape == (self.numMixtures, self.dim))
        assert(len(weights)== self.numMixtures)

        if means.dtype != np.float32:
            means = means.astype(np.float32)
        if diagCovs.dtype != np.float32:
            diagCovs = diagCovs.astype(np.float32)
        if weights.dtype != np.float32:
            weights = weights.astype(np.float32)


        #quick sanity checks
        self.means_gpu.set_async(means)
        self.diagCovs_gpu.set_async(diagCovs)
        self.weights_gpu.set(weights)

        self.likelihoodKernel.prepared_call((self.numBlocks, 1), (self.numThreads, 1, 1),
                                       self.Xpoints.gpudata, self.means_gpu.gpudata, self.diagCovs_gpu.gpudata,
                                       self.weights_gpu.gpudata,
                                       self.dim, self.numPoints, self.numMixtures,
                                       self.llVal.gpudata)

        ll = self.gpuarray.sum(self.llVal).get()
        return ll

try:
    import pycuda
    Likelihood = GPULL
except ImportError:
    try:
        from sklearn.mixture import GMM
        Likelihood= scikitLL
    except ImportError:
        Likelihood = SingleCoreLL

else:
    Likelihood = SingleCoreLL


def setup():
    import numpy as np
    a = np.random.random((100,1))
    sk = scikitLL(a, 4)
    sk.evaluator.fit(a)

    Sin = SingleCoreLL(a,4)

    print np.sum(sk.evaluator.score(a))

    print Sin.loglikelihood(sk.evaluator.means_, sk.evaluator.covars_, sk.evaluator.weights_)


if __name__ == '__main__':
    import timeit
    setupStr = """
import numpy as np
from __main__ import scikitLL, SingleCoreLL
a = np.random.random((100,1))
sk = scikitLL(a, 4)
sk.evaluator.fit(a)

Sin = SingleCoreLL(a,4)
    """
    actuSK = """
np.sum(sk.evaluator.score(a))
    """

    actuSin = """
Sin.loglikelihood(sk.evaluator.means_, sk.evaluator.covars_, sk.evaluator.weights_)
    """

    print timeit.timeit(actuSin, setupStr, number = 1000)
