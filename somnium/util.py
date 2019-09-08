def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    """
    Takes one or more iterable inputs as argument and splits it in equally-sized chunks.
    :param list_of_iterables: list of equally-sized iterables (list)
    :param n: chunk size (int)
    :param infinite: when set to True, the data is splited in chunks in a cycling way so that the iterator never ends
    (bool)
    :param return_incomplete_batches: if True, when n is not multiple of the length of the iterables, the last chunk
    is returned. Otherwise it is skipped (bool)
    :return: generator of iterables (list)
    """
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


def flatten(l):
    """
    Flattens a list of lists into a list
    :param l: list to flatten (list)
    :return:  list flattened (list)
    """
    return [item for sublist in l for item in sublist]
