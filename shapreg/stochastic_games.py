import numpy as np
from shapreg import utils


class StochasticCooperativeGame:
    '''Base class for stochastic cooperative games.'''

    def __init__(self):
        raise NotImplementedError

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

    def grand(self, batch_size):
        '''Get grand coalition value.'''
        N = 0
        mean_value = 0
        ones = np.ones((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(ones[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value

    def null(self, batch_size):
        '''Get null coalition value.'''
        N = 0
        mean_value = 0
        zeros = np.zeros((batch_size, self.players), dtype=bool)
        for U in self.iterate(batch_size):
            N += len(U[0])

            # Update mean value.
            value = self.__call__(zeros[:len(U[0])], U)
            mean_value += np.sum(value - mean_value, axis=0) / N

        return mean_value


class DatasetLossGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset.

    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      labels: array of corresponding labels.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, data, labels, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        assert len(labels) == self.N
        self.exogenous = (data, labels)

        # Store feature groups.
        num_features = data.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        '''
        Evaluate cooperative game.

        Args:
          S: array of player coalitions with size (batch, players).
          U: tuple of arrays of exogenous random variables, each with size
            (batch, dim).
        '''
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x, y = U

        # Possibly convert label datatype.
        if self.loss is utils.crossentropyloss:
            # Make sure not soft cross entropy.
            if (y.ndim == 1) or (y.shape[1] == 1):
                if np.issubdtype(y.dtype, np.floating):
                    y = y.astype(int)

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S), y)


class DatasetOutputGame(StochasticCooperativeGame):
    '''
    Cooperative game representing the model's loss over a dataset, with respect
    to the full model prediction.

    Args:
      extension: model extension (see removal.py).
      data: array of model inputs.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, data, loss, groups=None):
        self.extension = extension
        self.loss = loss
        self.N = len(data)
        self.exogenous = (data,)

        # Store feature groups.
        num_features = data.shape[1]
        if groups is None:
            self.players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.players = len(groups)
            self.groups_matrix = np.zeros(
                (len(groups), num_features), dtype=bool)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = True

    def __call__(self, S, U):
        '''
        Evaluate cooperative game.

        Args:
          S: array of player coalitions with size (batch, players).
          U: tuple of arrays of exogenous random variables, each with size
            (batch, dim).
        '''
        # Unpack exogenous random variable.
        if U is None:
            U = self.sample(len(S))
        x = U[0]

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(x, S),
                           self.extension(x, np.ones(x.shape, dtype=bool)))
