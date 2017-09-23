import numpy as np

CHUNK_SIZE = 'iters'
GRID_INDEX = 'index'


def _next_grid(x, grid_size):
    grid_idx = int(x // grid_size) + 1
    x_next = grid_size * grid_idx
    assert(x_next > x)
    assert(x_next - x <= grid_size)  # no skipping
    return grid_idx, x_next


def time_chunker(input_gen, grid_size, timers, keep_types=True, n_grid=np.inf):
    assert(grid_size > 0)
    # Possible to just pass a single timer
    if np.ndim(timers) == 0:
        timers = [('time', timers)]
    assert(np.ndim(timers) == 2 and np.shape(timers)[1] == 2)
    timer_names, __ = zip(*timers)
    assert(CHUNK_SIZE not in timer_names)
    assert(GRID_INDEX not in timer_names)

    dtype = object if keep_types else None
    primary = 0  # First timer is the primary, others extra

    total = 0
    idx = 0  # Note: this gives a stop the first time around
    chunk_size = 0
    # Could skip this and simplify if recompute next stop each iter
    next_stop = grid_size * idx
    # Preserve dtypes by initializing by a call
    chunk_timers = 0 * np.array([f() for _, f in timers], dtype=dtype)
    while idx < n_grid:  # StopIteration from input_gen will break loop too.
        start = [f() for _, f in timers]
        # If really concerned for accuracy, we could redo start[primary] here
        next_item = next(input_gen)
        finish = [f() for _, f in timers]

        # Would be slightly more elegant w/ pandas but not worth overhead
        deltas = np.array(finish, dtype=dtype) - np.array(start, dtype=dtype)
        # Allowing =0 for now, but that could have weird corner cases
        assert(deltas[primary] >= 0)
        total += deltas[primary]

        if total > next_stop:
            assert(idx == 0 or chunk_size > 0)  # Should be impossible
            # Prepare results of this reound. Note: this is the first item
            # after the time grid!
            timer_dict = dict(zip(timer_names, chunk_timers))
            timer_dict[CHUNK_SIZE] = chunk_size
            timer_dict[GRID_INDEX] = idx
            yield next_item, timer_dict

            # Setup next round
            chunk_timers *= 0
            chunk_size = 0
            idx, next_stop = _next_grid(total, grid_size)
        chunk_timers += deltas
        chunk_size += 1
