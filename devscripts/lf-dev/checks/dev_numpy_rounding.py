import numpy as np
from pprint import pprint


def main() -> None:
    a = np.array([0.1,0.5,0.7,1.0,1.25,1.5,1.75])
    print(a)

    round_method = np.round
    print(round_method(a))

if __name__ == '__main__':
    main()
