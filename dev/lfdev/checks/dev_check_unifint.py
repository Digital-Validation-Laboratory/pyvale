import numpy as np

def fun(x: float) -> float:
    return x**2

def main() -> None:
    x = np.linspace(0,2,5)
    h = 0.5

    print(x)

    f = fun(x)
    print(f)

    print(h*np.sum(f))
    print(np.sum(h*fun(x)))




if __name__ == "__main__":
    main()