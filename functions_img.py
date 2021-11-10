import cv2


def thumbnail_image(image_name, basewidth=None, path=None):
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
    image = cv2.imread(path + image_name, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]
    
    # Calculating the size to preserve aspect ratio
    r = basewidth / float(h)
    dim = (int(w*r), basewidth)
    
    # Resize image
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # Saving the new image
    cv2.imwrite(path + "thumbnails/" + image_name, image_resized)
    
    
    
    