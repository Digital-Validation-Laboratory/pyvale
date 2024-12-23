import numpy as np

def main() -> None:
    coeffs: np.ndarray = np.array([0,0,1])
    time: np.ndarray = np.linspace(0,10,11)
    print(time)

    poly_check: np.ndarray = np.zeros_like(time)
    for ii,cc in enumerate(coeffs):
        print(ii)
        print(cc)
        poly_check += cc * time ** ii
    
    print(poly_check)

if __name__ == "__main__":
    main()