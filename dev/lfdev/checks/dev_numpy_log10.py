import numpy as np
from pprint import pprint


def main() -> None:
    a = [110,1900]
    for aa in a:

        print(int(np.log10(aa)))

if __name__ == '__main__':
    main()
