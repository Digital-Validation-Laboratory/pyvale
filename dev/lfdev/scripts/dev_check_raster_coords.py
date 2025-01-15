

def main() -> None:
    near_clip_plane: float = 1.0
    far_clip_plane:float  = 1000.0

    image_width: int = 640
    image_height: int = 480

    inch_to_mm = 25.4
    film_aperture_width_inch = 0.980
    film_aperture_height_inch = 0.735

    focal_length = 20

    film_aspect_ratio = film_aperture_width_inch / film_aperture_height_inch
    device_aspect_ratio = float(image_width)/ float(image_height)

    top = ((film_aperture_height_inch*inch_to_mm/2) / focal_length) * near_clip_plane
    right = ((film_aperture_width_inch*inch_to_mm/2) / focal_length) * near_clip_plane

    print()
    print(80*"-")
    print(f"{film_aspect_ratio=}")
    print(f"{device_aspect_ratio=}")
    print()
    print(f"{top=}")
    print(f"{right=}")
    print(80*"-")
    print()


if __name__ == "__main__":
    main()