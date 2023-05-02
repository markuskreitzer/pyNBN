def dispf(AA, d1=9, d2=1, st=None):
    # dispf(AA, d1, d2) - display matrix as float
    #
    #    Arguments:
    #           AA -   matrix
    #           d1 -   length of number
    #           d2 -   number of decimal places

    if st is not None:
        str_ = f" <- {st}"
    else:
        str_ = " <= "

    np, ni = AA.shape
    sf = f"{{: {d1}.{d2}f}}"

    for i in range(np):
        if i < np - 1:
            print(" ".join([sf.format(x) for x in AA[i]]))
        else:
            print(" ".join([sf.format(x) for x in AA[i]]) + str_)

    return

