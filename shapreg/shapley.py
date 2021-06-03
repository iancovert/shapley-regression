import numpy as np
from shapreg import utils, games, stochastic_games
from tqdm.auto import tqdm


def default_min_variance_samples(game):
    '''Determine min_variance_samples.'''
    return 5


def default_variance_batches(game, batch_size):
    '''
    Determine variance_batches.

    This value tries to ensure that enough samples are included to make A
    approximation non-singular.
    '''
    if isinstance(game, games.CooperativeGame):
        return int(np.ceil(10 * game.players / batch_size))
    else:
        # Require more intermediate samples for stochastic games.
        return int(np.ceil(25 * game.players / batch_size))


def calculate_result(A, b, total):
    '''Calculate the regression coefficients.'''
    num_players = A.shape[1]
    try:
        if len(b.shape) == 2:
            A_inv_one = np.linalg.solve(A, np.ones((num_players, 1)))
        else:
            A_inv_one = np.linalg.solve(A, np.ones(num_players))
        A_inv_vec = np.linalg.solve(A, b)
        values = (
            A_inv_vec -
            A_inv_one * (np.sum(A_inv_vec, axis=0, keepdims=True) - total)
            / np.sum(A_inv_one))
    except np.linalg.LinAlgError:
        raise ValueError('singular matrix inversion. Consider using larger '
                         'variance_batches')

    return values


def ShapleyRegression(game,
                      batch_size=512,
                      detect_convergence=True,
                      thresh=0.01,
                      n_samples=None,
                      paired_sampling=True,
                      return_all=False,
                      min_variance_samples=None,
                      variance_batches=None,
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

    if min_variance_samples is None:
        min_variance_samples = default_min_variance_samples(game)
    else:
        assert isinstance(min_variance_samples, int)
        assert min_variance_samples > 1

    if variance_batches is None:
        variance_batches = default_variance_batches(game, batch_size)
    else:
        assert isinstance(variance_batches, int)
        assert variance_batches >= 1

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
    n = 0
    b = 0
    A = 0
    estimate_list = []

    # For variance estimation.
    A_sample_list = []
    b_sample_list = []

    # For tracking progress.
    var = np.nan * np.ones(num_players)
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
            A_sample = 0.5 * (
                np.matmul(S[:, :, np.newaxis].astype(float),
                          S[:, np.newaxis, :].astype(float))
                + np.matmul(np.logical_not(S)[:, :, np.newaxis].astype(float),
                            np.logical_not(S)[:, np.newaxis, :].astype(float)))
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
                    S.astype(float).T * game_eval[:, np.newaxis].T +
                    S_comp.astype(float).T * comp_eval[:, np.newaxis].T).T
        else:
            # Single sample.
            A_sample = np.matmul(S[:, :, np.newaxis].astype(float),
                                 S[:, np.newaxis, :].astype(float))
            if stochastic:
                b_sample = (S.astype(float).T
                            * (game(S, U) - null)[:, np.newaxis].T).T
            else:
                b_sample = (S.astype(float).T
                            * (game(S) - null)[:, np.newaxis].T).T

        # Welford's algorithm.
        n += batch_size
        b += np.sum(b_sample - b, axis=0) / n
        A += np.sum(A_sample - A, axis=0) / n

        # Calculate progress.
        values = calculate_result(A, b, total)
        A_sample_list.append(A_sample)
        b_sample_list.append(b_sample)
        if len(A_sample_list) == variance_batches:
            # Aggregate samples for intermediate estimate.
            A_sample = np.concatenate(A_sample_list, axis=0).mean(axis=0)
            b_sample = np.concatenate(b_sample_list, axis=0).mean(axis=0)
            A_sample_list = []
            b_sample_list = []

            # Add new estimate.
            estimate_list.append(calculate_result(A_sample, b_sample, total))

            # Estimate current var.
            if len(estimate_list) >= min_variance_samples:
                var = np.array(estimate_list).var(axis=0)

        # Convergence ratio.
        std = np.sqrt(var * variance_batches / (it + 1))
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
