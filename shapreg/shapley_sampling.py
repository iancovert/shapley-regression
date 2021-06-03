import numpy as np
from shapreg import utils, games, stochastic_games
from tqdm.auto import tqdm


def ShapleySampling(game,
                    batch_size=512,
                    detect_convergence=True,
                    thresh=0.01,
                    n_samples=None,
                    antithetical=False,
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

    # Calculate null coalition value.
    if stochastic:
        null = game.null(batch_size=batch_size)
    else:
        null = game.null()

    # Set up bar.
    n_loops = int(np.ceil(n_samples / batch_size))
    if bar:
        if detect_convergence:
            bar = tqdm(total=1)
        else:
            bar = tqdm(total=n_loops * batch_size)

    # Setup.
    num_players = game.players
    if isinstance(null, np.ndarray):
        values = np.zeros((num_players, len(null)))
        sum_squares = np.zeros((num_players, len(null)))
        deltas = np.zeros((batch_size, num_players, len(null)))
    else:
        values = np.zeros((num_players))
        sum_squares = np.zeros((num_players))
        deltas = np.zeros((batch_size, num_players))
    permutations = np.tile(np.arange(game.players), (batch_size, 1))
    arange = np.arange(batch_size)
    n = 0

    # For tracking progress.
    if return_all:
        N_list = []
        std_list = []
        val_list = []

    # Begin sampling.
    for it in range(n_loops):
        for i in range(batch_size):
            if antithetical and i % 2 == 1:
                permutations[i] = permutations[i - 1][::-1]
            else:
                np.random.shuffle(permutations[i])
        S = np.zeros((batch_size, game.players), dtype=bool)

        # Sample exogenous (if applicable).
        if stochastic:
            U = game.sample(batch_size)

        # Unroll permutations.
        prev_value = null
        for i in range(num_players):
            S[arange, permutations[:, i]] = 1
            if stochastic:
                next_value = game(S, U)
            else:
                next_value = game(S)
            deltas[arange, permutations[:, i]] = next_value - prev_value
            prev_value = next_value

        # Welford's algorithm.
        n += batch_size
        diff = deltas - values
        values += np.sum(diff, axis=0) / n
        diff2 = deltas - values
        sum_squares += np.sum(diff * diff2, axis=0)

        # Calculate progress.
        var = sum_squares / (n ** 2)
        std = np.sqrt(var)
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
            val_list.append(np.copy(values))
            std_list.append(np.copy(std))
            if detect_convergence:
                N_list.append(N_est)

    # Return results.
    if return_all:
        # Dictionary for progress tracking.
        iters = (np.arange(it + 1) + 1) * batch_size * num_players
        tracking_dict = {
            'values': val_list,
            'std': std_list,
            'iters': iters}
        if detect_convergence:
            tracking_dict['N_est'] = N_list

        return utils.ShapleyValues(values, std), tracking_dict
    else:
        return utils.ShapleyValues(values, std)
