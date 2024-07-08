import numpy as np

def main() -> None:
    a = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    print(a)
    print(a.reshape((9,1)))

if __name__ == "__main__":
    main()