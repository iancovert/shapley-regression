import pickle
import numpy as np
from shapreg import plotting


def crossentropyloss(pred, target):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])


def mseloss(pred, target):
    '''MSE loss that does not average across samples.'''
    if len(pred.shape) == 1:
        pred = pred[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
    return np.sum((pred - target) ** 2, axis=1)


class ShapleyValues:
    '''For storing and plotting Shapley values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def plot(self,
             feature_names=None,
             sort_features=True,
             max_features=np.inf,
             orientation='horizontal',
             error_bars=True,
             color='C0',
             title='Feature Importance',
             title_size=20,
             tick_size=16,
             tick_rotation=None,
             axis_label='',
             label_size=16,
             figsize=(10, 7),
             return_fig=False):
        '''
        Plot Shapley values.

        Args:
          feature_names: list of feature names.
          sort_features: whether to sort features by their Shapley values.
          max_features: number of features to display.
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          color: bar chart color.
          title: plot title.
          title_size: font size for title.
          tick_size: font size for feature names and numerical values.
          tick_rotation: tick rotation for feature names (vertical plots only).
          label_size: font size for label.
          figsize: figure size (if fig is None).
          return_fig: whether to return matplotlib figure object.
        '''
        return plotting.plot(
            self, feature_names, sort_features, max_features, orientation,
            error_bars, color, title, title_size, tick_size, tick_rotation,
            axis_label, label_size, figsize, return_fig)

    def comparison(self,
                   other_values,
                   comparison_names=None,
                   feature_names=None,
                   sort_features=True,
                   max_features=np.inf,
                   orientation='vertical',
                   error_bars=True,
                   colors=None,
                   title='Shapley Value Comparison',
                   title_size=20,
                   tick_size=16,
                   tick_rotation=None,
                   axis_label='',
                   label_size=16,
                   legend_loc=None,
                   figsize=(10, 7),
                   return_fig=False):
        '''
        Plot comparison with another set of Shapley values.

        Args:
          other_values: another Shapley values object.
          comparison_names: tuple of names for each Shapley value object.
          feature_names: list of feature names.
          sort_features: whether to sort features by their Shapley values.
          max_features: number of features to display.
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          colors: colors for each set of Shapley values.
          title: plot title.
          title_size: font size for title.
          tick_size: font size for feature names and numerical values.
          tick_rotation: tick rotation for feature names (vertical plots only).
          label_size: font size for label.
          legend_loc: legend location.
          figsize: figure size (if fig is None).
          return_fig: whether to return matplotlib figure object.
        '''
        return plotting.comparison_plot(
            (self, other_values), comparison_names, feature_names,
            sort_features, max_features, orientation, error_bars, colors, title,
            title_size, tick_size, tick_rotation, axis_label, label_size,
            legend_loc, figsize, return_fig)

    def save(self, filename):
        '''Save Shapley values object.'''
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')

    def __repr__(self):
        with np.printoptions(precision=2, threshold=12, floatmode='fixed'):
            return 'Shapley Values(\n  (Mean): {}\n  (Std):  {}\n)'.format(
                self.values, self.std)


def load(filename):
    '''Load Shapley values object.'''
    with open(filename, 'rb') as f:
        shapley_values = pickle.load(f)
        if isinstance(shapley_values, ShapleyValues):
            return shapley_values
        else:
            raise ValueError('object is not instance of ShapleyValues class')
