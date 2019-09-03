def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert (len({len(it) for it in list_of_iterables}) == 1)
    length = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, length, n):
            if not return_incomplete_batches:
                if (ndx + n) > length:
                    break
            yield [iterable[ndx:min(ndx + n, length)] for iterable in list_of_iterables]

        if not infinite:
            break