import cv2 as cv


def show_image(window_name: str, img: cv.typing.MatLike) -> None:
    """
    Show the image in the full screen window.

    :param window_name: The name of the window.
    :param img: The image to show.
    """

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(window_name, img)
    cv.waitKey(0)

    cv.destroyAllWindows()
