import numpy as np

def main() -> None:
    v1 = np.array([1,2,1])
    v2 = np.array([2,2,1])
    print(v1*v2)
    print(np.multiply(v1,v2))

    vs = np.array([[1,0,2],
                   [2,1,1]])

    test = np.apply_along_axis(np.multiply,0,v1,vs)
    print(test)

if __name__ == "__main__":
    main()