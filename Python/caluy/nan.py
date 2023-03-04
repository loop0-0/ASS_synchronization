import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)


def resize_image_to_show(image_to_prepare):
    max_photo_resolution_value = show_image_height
    
    old_image     = cv2.imread(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    max_val = max(actual_width, actual_height)

    if max_val >= max_photo_resolution_value:
        downscale_factor = max_val/max_photo_resolution_value
        new_width      = round(old_image.shape[1]/downscale_factor)
        new_height     = round(old_image.shape[0]/downscale_factor)
        resized_image  = cv2.resize(old_image,
                                (new_width, new_height),
                                interpolation = cv2.INTER_LINEAR)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        return image_to_prepare
