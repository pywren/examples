import scipy.linalg
import sklearn.metrics as metrics
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder
import time

def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    return y[:, 1]


def learnClassWeightedPrimal(trainData, labels, class_weights=None, 
                             reg=0.1, TOT_FEAT=1):
    '''Learn a model from trainData -> labels, with up weighted positive class '''

    num_classes = max(labels) + 1
    n = trainData.shape[0]
    if (class_weights is None):
        W = None
    else:
        W = np.ones(n)[:, np.newaxis]
        for i, c in enumerate(class_weights):
            W[np.where(labels == i), :] = class_weights[i]
    return learnPrimal(trainData, labels, W, reg=reg, TOT_FEAT=TOT_FEAT)


def learnPrimal(trainData, labels, W=None, reg=0.1, TOT_FEAT=1):
    '''Learn a model from trainData -> labels '''

    print "learn primal", "reg=", reg
    trainData = trainData.reshape(trainData.shape[0],-1)
    print "reshaping data done"
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData).reshape(trainData.shape[0], -1)
    print "now contiguous"

    if (W is None):
        W = np.ones(n, dtype=trainData.dtype)[:, np.newaxis]
        sqrtW = np.sqrt(W)
    else:
        sqrtW = np.sqrt(W)
        X = X * sqrtW
    print "starting dot", X.dtype
    t1 = time.time()
    XTWX = X.T.dot(X)
    t2 = time.time()
    print "X.T.dot(X) took {:3.1f} sec".format(t2-t1), XTWX.shape
    XTWX /= trainData.shape[1] # float(TOT_FEAT)
    idxes = np.diag_indices(XTWX.shape[0])
    XTWX[idxes] += reg
    # generate one-hot encoding
    y = np.eye(max(labels) + 1)[labels].astype(trainData.dtype)
    print "Computing X.T.dot(W*y)", W.dtype, y.dtype
    XTWy = X.T.dot(W * y)
    t1 = time.time()
    print "Going to solve", XTWX.dtype, XTWy.dtype
    model = scipy.linalg.solve(XTWX, XTWy)
    t2 = time.time()
    print "solve took {:3.1f} sec".format(t2-t1)
    return model

def learnPrimalMultiReg(trainData, labels, W=None, regs=tuple(), TOT_FEAT=1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData).reshape(trainData.shape[0], -1)
    if (W is None):
        W = np.ones(n)[:, np.newaxis].astype(trainData.dtype)
    else:
        sqrtW = np.sqrt(W)
        X *= sqrtW
    t1 = time.time()
    XTWX = X.T.dot(X)
    t2 = time.time()
    print "X.T.dot(X) took", t2-t1
    XTWX /= float(trainData.shape[1])
    idxes = np.diag_indices(XTWX.shape[0])
    y = np.eye(max(labels) + 1)[labels].astype(trainData.dtype)
    t1 = time.time()
    XTWy = X.T.dot(W * y)
    t2 = time.time()
    print "X.T.dot(W * y) took", t2-t1
    models = []
    if isinstance(regs, np.float64):
        regs = [regs]
    for reg in regs:
        XTWX_cp = XTWX.copy()
        XTWX_cp[idxes] += reg
        # generate one-hot encoding
        t1 = time.time()
        try:

            model = scipy.linalg.solve(XTWX_cp, XTWy)
        except scipy.linalg.LinAlgError as e:
            print "WARNING Matrix was singular"
            model = np.zeros((XTWX_cp.shape[1], XTWy.shape[1]))
        t2 = time.time()
        print "solving reg=", reg, "dtypes", XTWX_cp.dtype, XTWy.dtype, "took", t2-t1

        models.append(model)
    return models

def learnClassWeightedPrimalMultiReg(trainData, labels, class_weights=None, 
                                     reg=[0.1], TOT_FEAT=1):

    num_classes = max(labels) + 1
    n = trainData.shape[0]

    if class_weights is not None:
        W = np.ones(n)[:, np.newaxis]
        for i, c in enumerate(class_weights):
            W[np.where(labels == i), :] = class_weights[i]
    else:
        W = None

    return learnPrimalMultiReg(trainData, labels, W, regs=reg, 
                               TOT_FEAT=TOT_FEAT), reg


def median_impute(X):
    imp = Imputer(strategy="median")
    return imp.fit_transform(X)

def standard_scaler(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

def center_data(X):
    X -= np.mean(X, axis=1)[:,np.newaxis]
    return X

def trainAndEvaluateDualModel(KTrain, KTest, labelsTrain, labelsTest, reg=0.1):
    model = learnDual(KTrain,labelsTrain, reg=reg)
    predTrainWeights = evaluateDualModel(KTrain, model)
    predTestWeights = evaluateDualModel(KTest, model)
    return (predTrainWeights, predTestWeights)

def learnDual(gramMatrix, labels, reg=0.1, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    y = np.eye(max(labels) + 1)[labels]
    idxes = np.diag_indices(gramMatrix.shape[0])
    gramMatrix /= float(TOT_FEAT)
    gramMatrix[idxes] += (reg)
    model = scipy.linalg.solve(gramMatrix, y)
    gramMatrix[idxes] -= (reg)
    gramMatrix *= TOT_FEAT
    return model

def trainAndEvaluateClassWeightedPrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.0, class_weights=None):
    model = learnClassWeightedPrimal(XTrain, labelsTrain, reg=reg, class_weights=class_weights)
    yTrainHat = XTrain.dot(model)[:,1]
    yTestHat = XTest.dot(model)[:,1]
    return (yTrainHat, yTestHat)

def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.0, W=None):
    model = learnPrimal(XTrain, labelsTrain, reg=reg, W=W)
    yTrainHat = XTrain.dot(model)[:,1]
    yTestHat = XTest.dot(model)[:,1]
    return (yTrainHat, yTestHat)

def computeDistanceMatrix(XTest, XTrain):
    XTest = XTest.reshape(XTest.shape[0],-1)
    XTrain = XTrain.reshape(XTrain.shape[0],-1)
    XTest = make_contig(XTest)
    XTrain = make_contig(XTrain)
    XTrain_norms = (np.linalg.norm(XTrain, axis=1) ** 2)[:, np.newaxis]
    XTest_norms = (np.linalg.norm(XTest, axis=1) ** 2)[:, np.newaxis]
    K = XTest.dot(XTrain.T)
    K *= -2
    K += XTrain_norms.T
    K += XTest_norms
    return K

def make_contig(X):
    return np.ascontiguousarray(X).reshape(X.shape)

def computeLinearGramMatrix(XTest, XTrain):
    XTest = XTest.reshape(XTest.shape[0],-1)
    XTrain = XTrain.reshape(XTrain.shape[0],-1)
    XTest = make_contig(XTest)
    XTrain = make_contig(XTrain)
    return XTest.dot(XTrain.T)



def computeRBFGramMatrix(XTest, XTrain, gamma=1):
    XTest = XTest.reshape(XTest.shape[0],-1)
    XTrain = XTrain.reshape(XTrain.shape[0],-1)
    XTest = make_contig(XTest)
    XTrain = make_contig(XTrain)
    gamma = -1.0 * gamma
    return np.exp(gamma*computeDistanceMatrix(XTest, XTrain))

class PrimalModel(object):
    def __init__(self, reg, W=None, class_weights=None):
        self.reg = reg
        self.W = W
        self.class_weights=None

    def fit(self, XTrain, labelsTrain):
        #self.enc = OneHotEncoder()
        #self.enc.fit(labelsTrain)
        #labelsTrainOH = self.enc.transform(labelsTrain).todense()

        print "fitting"
        self.model =  learnClassWeightedPrimal(XTrain, labelsTrain, 
                                               class_weights=self.class_weights, 
                                               reg=self.reg)
        
    def predict_proba(self, XTest):
        yTestHat = XTest.dot(self.model)
        return yTestHat

    def config(self):
        return {'reg' : self.reg}

class MultiLeastSquares(object):
    def __init__(self, reg, W=None, class_weights=None):
        self.reg = reg

        self.W = W
        self.class_weights=None

    def fit(self, XTrain, labelsTrain):
        return self.multifit(self, XTrain, labelsTrain)

    def multifit(self, XTrain, labelsTrain):

        self.models, self.regs =  learnClassWeightedPrimalMultiReg(XTrain, labelsTrain, 
                                                       class_weights=self.class_weights, 
                                                       reg=self.reg)
        
        res = []
        for m, r in zip(self.models, self.regs):
            p = PrimalModel(r)
            p.model = m 
            res.append(p)
        return res

    def predict_proba_multi(self, XTest):
        yTestHat = [XTest.dot(m)[:, 1] for m in self.models]
        return yTestHat

    def get_configs(self):
        return [{'reg' : r} for r in self.regs]


class PrimalModelGaussianRandomFeatures(object):
    def __init__(self, reg, sigma=1.0, num_random_features=4096, W=None, class_weights=None):
        self.reg = reg
        self.W = W
        self.class_weights=None

        self.num_random_features = num_random_features
        self.sigma = sigma


    def fit(self, XTrain, labelsTrain):
        #self.enc = OneHotEncoder()
        #self.enc.fit(labelsTrain)
        #labelsTrainOH = self.enc.transform(labelsTrain).todense()
        self.random_phase = np.random.uniform(0, 2*np.pi, 
                                              size=self.num_random_features)
        self.random_matrix = np.random.randn(XTrain.shape[1], 
                                             self.num_random_features) * self.sigma
        self.model =  learnClassWeightedPrimal(self.lift(XTrain), labelsTrain, 
                                               class_weights=self.class_weights, 
                                               reg=self.reg)

    def lift(self, X):
        a = np.cos(X.dot(self.random_matrix) + self.random_phase)

        return a

    def predict_proba(self, XTest):
        yTestHat = self.lift(XTest).dot(self.model)
        return yTestHat[:, 1]

    def config(self):
        return {'reg' : self.reg, 'class_weights' : self.class_weights, 
                'num_random_features' : self.num_random_features, 
                'sigma' : self.sigma}


class MultiPrimalModelGaussianRandomFeatures(object):
    def __init__(self, reg, sigma=1.0, 
                 num_random_features=4096, W=None, class_weights=None):
        if isinstance(reg, np.float64):
            self.reg = [reg]

        self.W = W
        self.class_weights=None

        self.num_random_features = num_random_features
        self.sigma = sigma

    def multifit(self, XTrain, labelsTrain):
        #self.enc = OneHotEncoder()
        #self.enc.fit(labelsTrain)
        #labelsTrainOH = self.enc.transform(labelsTrain).todense()
        self.random_phase = np.random.uniform(0, 2*np.pi, 
                                              size=self.num_random_features)
        self.random_matrix = np.random.randn(XTrain.shape[1], 
                                             self.num_random_features) * self.sigma

        self.models, self.regs =  learnClassWeightedPrimalMultiReg(self.lift(XTrain), 
                                                        labelsTrain, 
                                                        reg = self.reg, 
                                                        class_weights=self.class_weights)
        

    def lift(self, X):
        a = np.cos(X.dot(self.random_matrix) + self.random_phase)

        return a

    def predict_proba_multi(self, XTest):
        lifted = self.lift(XTest)

        yTestHat = [lifted.dot(m)[:, 1] for m in self.models]
        return yTestHat

    def get_configs(self):
        return [{'reg' : r} for r in self.regs]
