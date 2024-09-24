import numpy as np

def main() -> None:
    a1 = np.array([[-np.sqrt(0.6),-np.sqrt(0.6),0],
                   [-np.sqrt(0.6),np.sqrt(0.6),0],
                   [np.sqrt(0.6),-np.sqrt(0.6),0],
                   [np.sqrt(0.6),np.sqrt(0.6),0],
                   [-np.sqrt(0.6),0,0],
                   [0,-np.sqrt(0.6),0],
                   [0,np.sqrt(0.6),0],
                   [np.sqrt(0.6),0,0],
                   [0,0,0]])


    w1 = np.array([[25/81],
                   [25/81],
                   [25/81],
                   [25/81],
                   [40/81],
                   [40/81],
                   [40/81],
                   [40/81],
                   [64/81]])

    w_25 = 25/81 * np.ones([4,6,1,30])
    w_40 = 40/81 * np.ones([4,6,1,30])
    w_64 = 64/81 * np.ones([1,6,1,30])
    w = np.vstack((w_25,w_40,w_64))

    print(f'{w_25.shape=}')
    print(f'{w_40.shape=}')
    print(f'{w_64.shape=}')
    print(f'{w.shape=}')

    test = np.ones((4,) + (6,1,30))
    print(test.shape)


if __name__ == "__main__":
    main()