import numpy as np
from shapreg import utils


class StochasticCooperativeGame:
    '''Base class for stochastic cooperative games with exogenous variable U.'''
    def __init__(self):
        pass

    def __call__(self, S, U):
        raise NotImplementedError

    def sample(self, samples):
        '''Sample exogenous random variable.'''
        inds = np.random.choice(self.N, size=samples)
        return tuple(arr[inds] for arr in self.exogenous)

    def iterate(self, batch_size):
        '''Iterate over values for exogenous random variable.'''
        ind = 0
        while ind < self.N:
            yield tuple(arr[ind:(ind + batch_size)] for arr in self.exogenous)
            ind += batch_size


class DatasetLossGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset.

    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      labels: array of corresponding labels.
      loss: loss function (see utils.py).
    '''
    def __init__(self, extension, data, labels, loss):
        self.extension = extension
        self.loss = loss
        self.players = data.shape[1]
        self.N = len(data)
        assert len(labels) == self.N
        self.exogenous = (data, labels)

    def __call__(self, S, U=None):
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x, y = U

        # Return scalar is single subset.
        single_eval = (S.ndim == 1)
        if single_eval:
            S = S[np.newaxis]

        # Possibly convert label datatype.
        if self.loss is utils.crossentropyloss:
            # Make sure not soft cross entropy.
            if (y.ndim == 1) or (y.shape[1] == 1):
                if np.issubdtype(y.dtype, np.floating):
                    y = y.astype(int)

        # Evaluate.
        output = - self.loss(self.extension(x, S), y)
        if single_eval:
            output = output[0]
        return output


class DatasetOutputGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset, with respect
    to the full model prediction.

    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      loss: loss function (see utils.py).
    '''
    def __init__(self, extension, data, loss):
        self.extension = extension
        self.loss = loss
        self.players = data.shape[1]
        self.N = len(data)
        self.exogenous = (data,)

    def __call__(self, S, U=None):
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x = U[0]

        # Return scalar is single subset.
        single_eval = (S.ndim == 1)
        if single_eval:
            S = S[np.newaxis]

        # Evaluate.
        output = - self.loss(self.extension(x, S),
                             self.extension(x, np.ones(x.shape, dtype=bool)))
        if single_eval:
            output = output[0]
        return output
