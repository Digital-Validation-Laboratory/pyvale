import numpy as np

def main() -> None:
    a = np.array([0,0,0,0])
    a[0:-1] = 1
    print(a)

    a = np.array([0,0,0,0])
    a[-1] = 1
    print(a)

if __name__ == "__main__":
    main()