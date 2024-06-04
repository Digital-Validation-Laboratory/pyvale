import numpy as np
from pprint import pprint


def main() -> None:
    pc_noise = 1.0
    n_bits = 8

    image = np.array([[0,0.5],[0,1]])*2**n_bits # 0 - 256
    pprint(image)

    noise = np.random.default_rng().standard_normal(image.shape) # -1 -> 1
    pprint(noise)

    noise_bits = noise*2**n_bits*pc_noise/100
    pprint(noise_bits)

    noisy_image = image + noise_bits
    pprint(noisy_image)


    final_image = np.array(noisy_image,dtype=np.uint8)
    pprint(final_image)

    noisy_image = image + 2^n_bits * pc_noise/100 * \
    np.random.default_rng().standard_normal(image.shape)

if __name__ == '__main__':
    main()
