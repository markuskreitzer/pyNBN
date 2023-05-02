from enum import Enum

class ActType(Enum):
    LINEAR = 0
    UNIPOLAR = 1
    BIPOLAR = 2
    BIPOLAR_ELLIOT = 3
    UNIPOLAR_ELLIOT = 4

def actFuncDer(n, net, act, gain):
    de = 0
    act_type = ActType(act[n])
    gain_n = gain[n]

    match act_type:
        case ActType.LINEAR:
            out = gain_n * net
            der = gain_n
        case ActType.UNIPOLAR:
            out = 1 / (1 + np.exp(-gain_n * net))
            der = gain_n * (1 - out) * out
        case ActType.BIPOLAR:
            out = np.tanh(gain_n * net)
            der = gain_n * (1 - out * out)
        case ActType.BIPOLAR_ELLIOT:
            out = gain_n * net / (1 + gain_n * np.abs(net))
            der = 1 / ((gain_n * np.abs(net) + 1) ** 2)
        case ActType.UNIPOLAR_ELLIOT:
            out = 2 * gain_n * net / (1 + gain_n * np.abs(net)) - 1
            der = 2 * gain_n / (gain_n * np.abs(net) + 1) ** 2

    der += de
    return out, der

