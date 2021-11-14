import matplotlib.pyplot as plt
import numpy as np
import cv2


def image_size(image_name, path):
    """
    Method used to get the image size

    Parameters:
    -----------------
        image_name (str): Image name
        path (str): Image path

    Returns:
    -----------------
        Return width x height

    """

    image = cv2.imread(path + image_name, cv2.IMREAD_UNCHANGED)

    # Return width x height
    return(image.shape[1], image.shape[0])


def thumbnail_image(image_name, basewidth, path):
    """
    Method used to create a thumbnail image

    Parameters:
    -----------------
        image_name (str): Image name
        basewidth (str): Image of thumbnail
        path (str): Image path

    Returns:
    -----------------
        Image saved in path + thumbnails

    """

    # reading the image and ist attributes
    image = cv2.imread(path + image_name)
    h, w = image.shape[:2]

    # Calculating the size to preserve aspect ratio
    r = basewidth / float(h)
    dim = (int(w*r), basewidth)

    # Resize image
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Saving the new image
    cv2.imwrite(path + "thumbnails/" + image_name, image_resized)


def contrast_and_brightness(image_name, path):
    """
    Method used to fit the contrast and brightness automatically

    Parameters:
    -----------------
        image_name (str): Image name
        path (str): Image path

    Returns:
    -----------------
        None.
        Image saved in path + thumbnails + contrast_and_brightness

    """

    clip_hist_percent = 0.3

    # Reading the image and ist attributes
    image = cv2.imread(path + image_name)

    # Reading grays in the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index-1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size-1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255/(maximum_gray-minimum_gray)
    beta = -minimum_gray * alpha

    image_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Saving the new image
    cv2.imwrite(path + "contrast_and_brightness/" +
                image_name, image_result)


def show_image_and_histogram(image_name, original_path,
                             edited_image_path):
    """
    Method used to show image and its histogram

    Parameters:
    -----------------
        image_name (str): Name of original image
        original_path (str): Path of original image
        edited_image_path (str): Path of edited image

    Returns:
    -----------------
        None
        Plot original and edited image with their histograms

    """

    original_image = cv2.imread(original_path + image_name)
    edited_image = cv2.imread(edited_image_path + image_name)

    fig = plt.figure(figsize=(12, 8))

    ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), \
        fig.add_subplot(223), fig.add_subplot(224)

    ax1.imshow(original_image)
    ax1.set_title("Original image", fontsize=14)
    ax1.grid(None)
    ax1.axis("off")

    ax2.hist(np.array(original_image).flatten(), bins=range(256),
             facecolor="#2ab0ff", edgecolor="#169acf", linewidth=0.5)
    ax2.set_title("Histogram", fontsize=14)

    ax3.imshow(edited_image)
    ax3.set_title("Image after preprocessing", fontsize=14)
    ax3.grid(None)
    ax3.axis("off")

    ax4.hist(np.array(edited_image).flatten(), bins=range(256),
             facecolor="#2BDC6C", edgecolor="#0CD355", linewidth=0.5)
    ax4.set_title("Histogram after preprocessing", fontsize=14)

    plt.tight_layout()
    plt.show()


def gray_image(image_name, path):
    """
    Method used to transform image to gray

    Parameters:
    -----------------
        image_name (str): Name of original image
        path (str): Image path

    Returns:
    -----------------
        None.
        Image transform to gray

    """

    image = cv2.imread(path + image_name)

    image_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Saving the new image
    cv2.imwrite(path + "gray_images/" +
                image_name, image_result)


def noise_reduction(image_name, path):
    """
    Method used to reduce noise in image

    Parameters:
    -----------------
        image_name (str): Name of original image
        path (str): Image path

    Returns:
    -----------------
        None.
        Image with noise reduced

    """

    image = cv2.imread(path + image_name)

    image_result = cv2.fastNlMeansDenoising(image, h=3)

    # Saving the new image
    cv2.imwrite(path + "noise_reduction/" +
                image_name, image_result)


def plot_two_images(image_a, image_b,
                    title_a=None, title_b=None):
    """
    Method used to plot two images

    Parameters:
    -----------------
        image_a (img): Image on the left
        image_b (img): Image on the right

    Returns:
    -----------------
        None.
        Plot images

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image_a)
    ax1.grid(None)
    ax1.axis("off")
    if title_a is not None:
        ax1.set_title(title_a, fontsize=14)
    else:
        ax1.set_title("image a", fontsize=14)

    ax2.imshow(image_b)
    ax2.grid(None)
    ax2.axis("off")
    if title_a is not None:
        ax2.set_title(title_b, fontsize=14)
    else:
        ax2.set_title("image b", fontsize=14)

    plt.tight_layout()
    plt.show()
