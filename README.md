# Shapley Regression

This repository provides a linear regression-based estimator for the Shapley values of any cooperative game. The methods implemented here are described in [this paper](https://arxiv.org/abs/2012.01536). Although the code can be used for arbitrary cooperative games, our focus is model explanation methods such [SHAP](https://arxiv.org/abs/1705.07874), [SAGE](https://arxiv.org/abs/2004.00668), and [Shapley Effects](https://epubs.siam.org/doi/pdf/10.1137/130936233?casa_token=w_mumFZVCBoAAAAA:MW_cuyNDkRJhjnA-0OxIO56iRP706V7RwH9sLf_eguYd-91lqkR9mUBRj6TWoPFI9Ix_D34onp4).

Shapley values are already supported by several other repositories, so the goal of this package is to provide a fast linear regression-based estimator with the following features:

1. **Convergence detection.** The estimator will stop automatically when it is approximately converged, so you don't need to specify the number of samples.

2. **Convergence forecasting.** For use cases that take a long time to run, the estimator will forecast the amount of time required to reach convergence (displayed with a progress bar).

3. **Uncertainty estimates.** Shapley values are often estimated rather than calculated exactly, and our method provides confidence intervals for the results.

## Usage

To install the code, please clone this repository and install the module into your Python environment:

```bash
pip install .
```

Next, to run the code you only need to do two things: (i) specify a cooperative game, and (ii) run the Shapley value estimator. For example, you can calculate SHAP values as follows:

```python
from shapreg import removal, games, shapley

# Get data
x, y = ...
feature_names = ...

# Get model (callable)
model = ...

# Set up cooperative game (SHAP)
extension = removal.MarginalExtension(x[:128], model)
instance = x[0]
game = games.PredictionGame(extension, instance)

# Estimate Shapley values
values = shapley.ShapleyRegression(game)
values.plot(feature_names)
```

For usage examples, see the following notebooks:

- [Census](https://github.com/iancovert/shapley-regression/blob/master/notebooks/census.ipynb) shows how to explain individual predictions (SHAP)
- [Credit](https://github.com/iancovert/shapley-regression/blob/master/notebooks/credit.ipynb) shows how to explain the model's loss (SAGE)
- [Bank](https://github.com/iancovert/shapley-regression/blob/master/notebooks/bank.ipynb) shows how to explain the model's global sensitivity (Shapley Effects)

<!-- To replicate any experiments from the paper, see our experiments [here](https://github.com/iancovert/shapley-regression/blob/master/experiments). -->

# Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Su-In Lee

## References

Ian Covert and Su-In Lee. "Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression." *arxiv preprint:2012.01536*
