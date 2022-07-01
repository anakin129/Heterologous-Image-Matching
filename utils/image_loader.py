import cv2
def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        # convert to rgb and return
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None