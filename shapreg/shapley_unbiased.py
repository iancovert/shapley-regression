import numpy as np
from shapreg import utils, games, stochastic_games
from tqdm.auto import tqdm


def calculate_A(num_players):
    '''Calculate A parameter's exact form.'''
    p_coaccur = (
        (np.sum((np.arange(2, num_players) - 1)
                / (num_players - np.arange(2, num_players)))) /
        (num_players * (num_players - 1) *
         np.sum(1 / (np.arange(1, num_players)
                * (num_players - np.arange(1, num_players))))))
    A = np.eye(num_players) * 0.5 + (1 - np.eye(num_players)) * p_coaccur
    return A


def calculate_exact_result(A, b, total, b_sum_squares, n):
    '''Calculate the regression coefficients and uncertainty estimates.'''
    num_players = A.shape[1]
    scalar_values = len(b.shape) == 1
    if scalar_values:
        A_inv_one = np.linalg.solve(A, np.ones(num_players))
    else:
        A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
    A_inv_vec = np.linalg.solve(A, b)
    values = (
        A_inv_vec -
        A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total)
        / np.sum(A_inv_one))

    # Calculate variance.
    try:
        # Symmetrize, rescale and swap axes.
        b_sum_squares = 0.5 * (b_sum_squares
                               + np.swapaxes(b_sum_squares, 0, 1))
        b_cov = b_sum_squares / (n ** 2)
        if scalar_values:
            b_cov = np.expand_dims(b_cov, 0)
        else:
            b_cov = np.swapaxes(b_cov, 0, 2)

        # Use Cholesky decomposition to calculate variance.
        cholesky = np.linalg.cholesky(b_cov)
        L = (
            np.linalg.solve(A, cholesky) +
            np.matmul(np.outer(A_inv_one, A_inv_one), cholesky)
            / np.sum(A_inv_one))
        beta_cov = np.matmul(L, np.moveaxis(L, -2, -1))
        var = np.diagonal(beta_cov, offset=0, axis1=-2, axis2=-1)
        std = var ** 0.5
        if scalar_values:
            std = std[0]
        else:
            std = std.T
    except np.linalg.LinAlgError:
        # b_cov likely is not PSD due to insufficient samples.
        std = np.ones(num_players) * np.nan

    return values, std


def ShapleyRegression(game,
                      batch_size=512,
                      detect_convergence=True,
                      thresh=0.01,
                      n_samples=None,
                      paired_sampling=True,
                      return_all=False,
                      bar=True,
                      verbose=False):
    # Verify arguments.
    if isinstance(game, games.CooperativeGame):
        stochastic = False
    elif isinstance(game, stochastic_games.StochasticCooperativeGame):
        stochastic = True
    else:
        raise ValueError('game must be CooperativeGame or '
                         'StochasticCooperativeGame')

    # Possibly force convergence detection.
    if n_samples is None:
        n_samples = 1e20
        if not detect_convergence:
            detect_convergence = True
            if verbose:
                print('Turning convergence detection on')

    if detect_convergence:
        assert 0 < thresh < 1

    # Weighting kernel (probability of each subset size).
    num_players = game.players
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)

    # Calculate null and grand coalitions for constraints.
    if stochastic:
        null = game.null(batch_size=batch_size)
        grand = game.grand(batch_size=batch_size)
    else:
        null = game.null()
        grand = game.grand()

    # Calculate difference between grand and null coalitions.
    total = grand - null

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

    # Setup.
    A = calculate_A(num_players)
    n = 0
    b = 0
    b_sum_squares = 0

    # For tracking progress.
    if return_all:
        N_list = []
        std_list = []
        val_list = []

    # Begin sampling.
    for it in range(n_loops):
        # Sample subsets.
        S = np.zeros((batch_size, num_players), dtype=bool)
        num_included = np.random.choice(num_players - 1, size=batch_size,
                                        p=weights) + 1
        for row, num in zip(S, num_included):
            inds = np.random.choice(num_players, size=num, replace=False)
            row[inds] = 1

        # Sample exogenous (if applicable).
        if stochastic:
            U = game.sample(batch_size)

        # Update estimators.
        if paired_sampling:
            # Paired samples.
            if stochastic:
                game_eval = game(S, U) - null
                S_comp = np.logical_not(S)
                comp_eval = game(S_comp, U) - null
                b_sample = 0.5 * (
                    S.astype(float).T * game_eval[:, np.newaxis].T
                    + S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
            else:
                game_eval = game(S) - null
                S_comp = np.logical_not(S)
                comp_eval = game(S_comp) - null
                b_sample = 0.5 * (
                    S.astype(float).T * game_eval[:, np.newaxis].T
                    + S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
        else:
            # Single sample.
            if stochastic:
                b_sample = (S.astype(float).T
                            * (game(S, U) - null)[:, np.newaxis].T).T
            else:
                b_sample = (S.astype(float).T
                            * (game(S) - null)[:, np.newaxis].T).T

        # Welford's algorithm.
        n += batch_size
        b_diff = b_sample - b
        b += np.sum(b_diff, axis=0) / n
        b_diff2 = b_sample - b
        b_sum_squares += np.sum(
            np.expand_dims(b_diff, 2) * np.expand_dims(b_diff2, 1),
            axis=0)

        # Calculate progress.
        values, std = calculate_exact_result(A, b, total, b_sum_squares, n)
        ratio = np.max(
            np.max(std, axis=0) / (values.max(axis=0) - values.min(axis=0)))

        # Print progress message.
        if verbose:
            if detect_convergence:
                print(f'StdDev Ratio = {ratio:.4f} (Converge at {thresh:.4f})')
            else:
                print(f'StdDev Ratio = {ratio:.4f}')

        # Check for convergence.
        if detect_convergence:
            if ratio < thresh:
                if verbose:
                    print('Detected convergence')

                # Skip bar ahead.
                if bar:
                    bar.n = bar.total
                    bar.refresh()
                break

        # Forecast number of iterations required.
        if detect_convergence:
            N_est = (it + 1) * (ratio / thresh) ** 2
            if bar and not np.isnan(N_est):
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
        elif bar:
            bar.update(batch_size)

        # Save intermediate quantities.
        if return_all:
            val_list.append(values)
            std_list.append(std)
            if detect_convergence:
                N_list.append(N_est)

    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        iters = (
            (np.arange(it + 1) + 1) * batch_size *
            (1 + int(paired_sampling)))
        tracking_dict = {
            'values': val_list,
            'std': std_list,
            'iters': iters}
        if detect_convergence:
            tracking_dict['N_est'] = N_list

        return utils.ShapleyValues(values, std), tracking_dict
    else:
        return utils.ShapleyValues(values, std)
