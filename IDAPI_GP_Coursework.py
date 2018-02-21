import numpy as np
from scipy.optimize import minimize

from numpy import exp
from numpy import dot
from numpy import log
from numpy import pi
from numpy import trace

from numpy.linalg import cholesky
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import det

# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 2:
    # TODO: Implement a draw from a multivariate Gaussian here

    x = np.random.randn(mean.shape[0])
    sample = np.dot(cholesky(cov), x) + mean
    
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            # append the vector Xa to the end of X
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n, n))

        # Task 1:
        # TODO: Implement the covariance matrix here

        print ("X shape: ", X.shape)

        for p in range(n):
            for q in range(n):
                covMat[p][q] = self.k(X[p], X[q])

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat
    
    def k(self, xp, xq):
        params = self.getParamsExp()
        sigma2_f = params[0]
        length = params[1]

        return sigma2_f * exp(-(norm(xp-xq)**2) / (2*length**2))

class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

        print ("GPR X: ", X.shape)
        print ("GPR y: ", y.shape)
        print ("GPR k: ", k.getParamsExp())
        print ("GPR K: ", self.K.shape)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction

        print ("Xa shape: ", Xa.shape)

        ker_test = self.kerMatrix(Xa, self.X)
        mean_fa = dot(dot(ker_test, inv(self.K)), self.y)

        cov_fa = self.kerMatrix(Xa, Xa)
        cov_fa -= dot(dot(ker_test, inv(self.K)), self.kerMatrix(self.X, Xa))
        # Return the mean and covariance
        return mean_fa, cov_fa

    def kerMatrix(self, X, Xa):
        params = self.k.getParamsExp()
        sigma2_f = params[0]
        length = params[1]

        kerMat = np.zeros((X.shape[0], Xa.shape[0]))
        for p in range(X.shape[0]):
            for q in range(Xa.shape[0]):
                kerMat[p][q] = sigma2_f * exp(-(norm(X[p]-Xa[q])**2) / (2*length**2))
                
        return kerMat
    
    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y

        mll = dot(dot(self.y.T, inv(self.K)), self.y) / 2
        mll += log(det(self.K))/2 + self.y.shape[0]*log(2*pi)/2
        
        # Return mll
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters

        params = self.k.getParams()
        ln_sigma_f = params[0]
        ln_length = params[1]
        ln_sigma_n = params[2]

        grad_k_lnsigmaf = np.zeros_like(self.K)
        grad_k_lnlength = np.zeros_like(self.K)
        grad_k_lnsigman = np.zeros_like(self.K)
        n = self.X.shape[0]
        X = self.X
        for p in range(n):
            for q in range(n):
                temp = exp(2*ln_sigma_f - norm(X[p]-X[q])**2/(2*exp(2*ln_length)))
                grad_k_lnsigmaf[p][q] = 2 * temp
                grad_k_lnlength[p][q] = temp * norm(X[p]-X[q])**2 * exp(-2*ln_length)
                grad_k_lnsigman[p][q] = 2 * exp(2*ln_sigma_n) if p==q else 0.0

        grad_ln_sigma_f = self.gradTemplate(grad_k_lnsigmaf)
        grad_ln_length_scale = self.gradTemplate(grad_k_lnlength)
        grad_ln_sigma_n = self.gradTemplate(grad_k_lnsigman)
                
        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    def gradTemplate(self, grad):
        template = dot(self.y.T, inv(self.K))
        template = dot(template, grad)
        template = dot(dot(template, inv(self.K)), self.y)
        template -= trace(dot(inv(self.K), grad)) / 2
        return template
    
    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar

        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya

        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)
    
    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################

    # print (multivariateGaussianDraw(np.asarray([0,0]), np.asarray([[0.5, 0.5], [0.5, 0.5]])))

    # 2nd Question
    rbf = RadialBasisFunction([0, 1, 2])
    A = np.asarray([[1,2], [2,3]])
    print (rbf.covMatrix(A))

    X = np.asarray([[1,2], [2,3]])
    y = np.asarray([[1], [2]])
    Xt = np.asarray([[4, 5], [6, 7]])
    reg = GaussianProcessRegression(X, y, rbf)
    mean_fa, cov_fa = reg.predict(Xt)
    print (mean_fa, "\n \n", cov_fa)

    reg.logMarginalLikelihood()
    reg.gradLogMarginalLikelihood()
