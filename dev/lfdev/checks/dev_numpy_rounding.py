import numpy as np
from pprint import pprint


def main() -> None:
    a = np.linspace(-2.0,2.0,20)
    print(a)

    b = a
    b[b>1] = 1
    print(b)
    b[b<-1] = -1
    print(b)

if __name__ == '__main__':
    main()
