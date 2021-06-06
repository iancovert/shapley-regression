# Shapley Regression

This repository implements a regression-based approach to estimating Shapley values. Although the code can be used with any cooperative game, our focus is model explanation methods such [SHAP](https://arxiv.org/abs/1705.07874), [SAGE](https://arxiv.org/abs/2004.00668), and [Shapley Effects](https://epubs.siam.org/doi/pdf/10.1137/130936233?casa_token=w_mumFZVCBoAAAAA:MW_cuyNDkRJhjnA-0OxIO56iRP706V7RwH9sLf_eguYd-91lqkR9mUBRj6TWoPFI9Ix_D34onp4), which are the Shapley values of several specific cooperative games. The methods provided here were developed in [this paper](https://arxiv.org/abs/2012.01536).

Because approximations are essential in most practical Shapley value applications, we provide an estimation approach with the following convenient features:

1. **Convergence detection:** the estimator stops automatically when it is approximately converged, so you don't need to specify the number of samples.

2. **Convergence forecasting:** for use cases that take a long time to run, our implementation forecasts the amount of time required to reach convergence (displayed with a progress bar).

3. **Uncertainty estimation:** Shapley values are often estimated rather than calculated exactly, and our method provides confidence intervals for the results.

## Usage

To use the code, clone this repository and install the package into your Python environment:

```bash
pip install .
```

Next, to run the code you only need to do two things: 1) specify a cooperative game, and 2) run the Shapley value estimator. For example, you can calculate SHAP values as follows:

```python
from shapreg import removal, games, shapley

# Get data
x, y = ...
feature_names = ...

# Get model (a callable object)
model = ...

# Set up the cooperative game (SHAP)
imputer = removal.MarginalExtension(x[:128], model)
game = games.PredictionGame(imputer, x[0])

# Estimate Shapley values
values = shapley.ShapleyRegression(game)
```

For examples, see the following notebooks:

- [Census](https://github.com/iancovert/shapley-regression/blob/master/notebooks/census.ipynb): shows how to explain individual predictions (SHAP)
- [Credit](https://github.com/iancovert/shapley-regression/blob/master/notebooks/credit.ipynb): shows how to explain the model's loss (SAGE)
- [Bank](https://github.com/iancovert/shapley-regression/blob/master/notebooks/bank.ipynb): shows how to explain the model's global sensitivity (Shapley Effects)
- [Consistency](https://github.com/iancovert/shapley-regression/blob/master/notebooks/consistency.ipynb): verifies that our different estimators return the same results
- [Calibration](https://github.com/iancovert/shapley-regression/blob/master/notebooks/calibration.ipynb): verifies the accuracy of our uncertainty estimates


# Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Su-In Lee

## References

Ian Covert and Su-In Lee. "Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression." *arxiv preprint:2012.01536*

