import numpy as np
from shapreg import utils


class CooperativeGame:
    '''Base class for cooperative games.'''

    def __init__(self):
        raise NotImplementedError

    def __call__(self, S):
        '''Evaluate cooperative game.'''
        raise NotImplementedError

    def grand(self):
        '''Get grand coalition value.'''
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        '''Get null coalition value.'''
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]


class PredictionGame(CooperativeGame):
    '''
    Cooperative game for an individual example's prediction.

    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
    '''

    def __init__(self, extension, sample, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]
        elif sample.shape[0] != 1:
            raise ValueError('sample must have shape (ndim,) or (1,ndim)')

        self.extension = extension
        self.sample = sample

        # Store feature groups.
        num_features = sample.shape[1]
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

        # Caching.
        self.sample_repeat = sample

    def __call__(self, S):
        '''
        Evaluate cooperative game.

        Args:
          S: array of player coalitions with size (batch, players).
        '''
        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
        input_data = self.sample_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return self.extension(input_data, S)


class PredictionLossGame(CooperativeGame):
    '''
    Cooperative game for an individual example's loss value.

    Args:
      extension: model extension (see removal.py).
      sample: numpy array representing a single model input.
      label: the input's true label.
      loss: loss function (see utils.py).
    '''

    def __init__(self, extension, sample, label, loss, groups=None):
        # Add batch dimension to sample.
        if sample.ndim == 1:
            sample = sample[np.newaxis]

        # Add batch dimension to label.
        if np.isscalar(label):
            label = np.array([label])

        # Convert label dtype if necessary.
        if loss is utils.crossentropyloss:
            # Make sure not soft cross entropy.
            if (label.ndim <= 1) or (label.shape[1] == 1):
                # Only convert if float.
                if np.issubdtype(label.dtype, np.floating):
                    label = label.astype(int)

        self.extension = extension
        self.sample = sample
        self.label = label
        self.loss = loss

        # Store feature groups.
        num_features = sample.shape[1]
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

        # Caching.
        self.sample_repeat = sample
        self.label_repeat = label

    def __call__(self, S):
        '''
        Evaluate cooperative game.

        Args:
          S: array of player coalitions with size (batch, players).
        '''
        # Try to use caching for repeated data.
        if len(S) != len(self.sample_repeat):
            self.sample_repeat = self.sample.repeat(len(S), 0)
            self.label_repeat = self.label.repeat(len(S), 0)
        input_data = self.sample_repeat
        output_label = self.label_repeat

        # Apply group transformation.
        if self.groups_matrix is not None:
            S = np.matmul(S, self.groups_matrix)

        # Evaluate.
        return - self.loss(self.extension(input_data, S), output_label)


# class DatasetLossGame(CooperativeGame):
#     '''
#     Cooperative game representing the model's loss over a dataset.

#     Args:
#       extension: model extension (see removal.py).
#       data: array of model inputs.
#       labels: array of corresponding labels.
#       loss: loss function (see utils.py).
#     '''

#     def __init__(self, extension, data, labels, loss, groups=None):
#         # Convert labels dtype if necessary.
#         if loss is utils.crossentropyloss:
#             # Make sure not soft cross entropy.
#             if (labels.ndim == 1) or (labels.shape[1] == 1):
#                 # Only convert if float.
#                 if np.issubdtype(labels.dtype, np.floating):
#                     labels = labels.astype(int)

#         self.extension = extension
#         self.data = data
#         self.labels = labels
#         self.loss = loss

#         # Store feature groups.
#         num_features = data.shape[1]
#         if groups is None:
#             self.players = num_features
#             self.groups_matrix = None
#         else:
#             # Verify groups.
#             inds_list = []
#             for group in groups:
#                 inds_list += list(group)
#             assert np.all(np.sort(inds_list) == np.arange(num_features))

#             # Map groups to features.
#             self.players = len(groups)
#             self.groups_matrix = np.zeros(
#                 (len(groups), num_features), dtype=bool)
#             for i, group in enumerate(groups):
#                 self.groups_matrix[i, group] = True

#         # Caching.
#         self.data_tile = self.data
#         self.label_tile = self.labels

#     def __call__(self, S):
#         '''
#         Evaluate cooperative game.

#         Args:
#           S: array of player coalitions with size (batch, players).
#         '''
#         # Prepare data.
#         if len(self.data_tile) != len(self.data) * len(S):
#             self.data_tile = np.tile(self.data, (len(S), 1))
#             self.label_tile = np.tile(
#                 self.labels,
#                 (len(S), *[1 for _ in range(len(self.labels.shape[1:]))]))
#         S = S.repeat(len(self.data), 0)

#         # Apply group transformation.
#         if self.groups_matrix is not None:
#             S = np.matmul(S, self.groups_matrix)

#         # Evaluate.
#         output = - self.loss(self.extension(self.data_tile, S), self.label_tile)
#         output = output.reshape((-1, self.data.shape[0]))
#         return np.mean(output, axis=1)


# class DatasetOutputGame(CooperativeGame):
#     '''
#     Cooperative game representing the model's loss over a dataset, with respect
#     to the full model prediction.

#     Args:
#       extension: model extension (see removal.py).
#       data: array of model inputs.
#       loss: loss function (see utils.py).
#     '''

#     def __init__(self, extension, data, loss, groups=None):
#         self.extension = extension
#         self.data = data
#         self.loss = loss
#         self.data_tile = self.data

#         # Store feature groups.
#         num_features = data.shape[1]
#         if groups is None:
#             self.players = num_features
#             self.groups_matrix = None
#         else:
#             # Verify groups.
#             inds_list = []
#             for group in groups:
#                 inds_list += list(group)
#             assert np.all(np.sort(inds_list) == np.arange(num_features))

#             # Map groups to features.
#             self.players = len(groups)
#             self.groups_matrix = np.zeros(
#                 (len(groups), num_features), dtype=bool)
#             for i, group in enumerate(groups):
#                 self.groups_matrix[i, group] = True

#         # Create labels.
#         self.labels = extension(data, np.ones(data.shape, dtype=bool))
#         self.label_tile = self.labels

#     def __call__(self, S):
#         '''
#         Evaluate cooperative game.

#         Args:
#           S: array of player coalitions with size (batch, players).
#         '''
#         # Prepare data.
#         if len(self.data_tile) != len(self.data) * len(S):
#             self.data_tile = np.tile(self.data, (len(S), 1))
#             self.label_tile = np.tile(
#                 self.labels,
#                 (len(S), *[1 for _ in range(len(self.labels.shape[1:]))]))
#         S = S.repeat(len(self.data), 0)

#         # Apply group transformation.
#         if self.groups_matrix is not None:
#             S = np.matmul(S, self.groups_matrix)

#         # Evaluate.
#         output = - self.loss(self.extension(self.data_tile, S), self.label_tile)
#         output = output.reshape((-1, self.data.shape[0]))
#         return np.mean(output, axis=1)
