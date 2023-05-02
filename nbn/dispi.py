from typing import Optional
import numpy as np

def dispi(AA: np.ndarray, d: int = 3, st: Optional[str] = None) -> None:
    # dispi(AA, d) - display matrix as integers
    #
    #    Arguments:
    #           AA -   matrix
    #           d -   length of number
    #           string to display

    if st is not None:
        str_ = f" <- {st}"
    else:
        str_ = " <= "

    np_, ni = AA.shape
    AA = np.round(AA).astype(int)
    sf = f"{{: {d}d}}"

    for i in range(np_):
        if i < np_ - 1:
            print(" ".join([sf.format(x) for x in AA[i]]))
        else:
            print(" ".join([sf.format(x) for x in AA[i]]) + str_)

    return

