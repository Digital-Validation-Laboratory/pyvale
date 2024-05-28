import numpy as np
from pprint import pprint

def ret_tuple() -> tuple[int,int,int]:
    return (1,2,3)

def main() -> None:
    (test,_,_) = ret_tuple()
    print(test)

if __name__ == '__main__':
    main()
