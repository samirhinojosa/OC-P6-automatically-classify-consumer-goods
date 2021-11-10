import cv2


def image_resize(image, width=None, height=None, inter):
    
    
    
    
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def thumbnail_image(image):
    
    basewidth = 224
    
    # Calculating the new size considering its aspect ratio
    width = (basewidth/float(image.shape[1]))
    height = int((float(image.shape[0])*float(wpercent)))
    dim = (width, height)
    
    # Resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # Saving new image
    cv2.imwrite(ORIGINAL_IMAGES_PATH + "thumbnail", resized) 
    
    
    
    