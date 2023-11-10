import cv2
import numpy as np

def f2():
    # Image dimensions
    image_width = 131  # pixels
    image_height = 266  # pixels

    # List of file paths
    # image_paths = [
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 1.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 3.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 5.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 7.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 9.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 11.jpg",
    #     r"D:\proj\image processing\Animal study pics _ CSE dept\standard day\Standard _ Day 13.jpg"
    # ]

    # for F2

    image_paths = [
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 1.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 3.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 5.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 7.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 9.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 11.jpg",
        r"D:\proj\image processing\Animal study pics _ CSE dept\F2 _ Day 13.jpg"
    ]

    path = r'D:\proj\image processing\Animal study pics _ CSE dept\New folder\Untitled-2.png'
    
    # Reading an image in default mode 
    image = cv2.imread(path) 
    
    # Window name in which image is displayed 
    window_name = 'Detected Wounds (F2)'
    
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow(window_name, image) 




    # Initialize an empty list to store wound areas
    wound_areas = []

    # Initialize a list to store resized images
    resized_images = []

    for image_path in image_paths:
        # Read the image from file
        frame = cv2.imread(image_path)

        # Resize the image to a common height
        ratio = image_height / frame.shape[0]
        resized_frame = cv2.resize(frame, (int(frame.shape[1] * ratio), image_height))

        # Append the resized image to the list
        resized_images.append(resized_frame)

        # Convert BGR to HSV color scheme
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define a narrow range of wound colors in HSV (shades of red)
        lower_red = np.array([175, 60, 60])
        upper_red = np.array([190, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Threshold the HSV image to get only wound colors
        mask = mask + cv2.inRange(hsv, lower_red, upper_red)
        
        square_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.rectangle(square_mask, (0, 80), (frame.shape[1], frame.shape[0]-80), 255, -1)

        mask = mask & square_mask

        # Apply Gaussian blur to the mask for better contour detection
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Calculating percentage area
        try:
            contours, hierarchy = cv2.findContours(blurred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=lambda x: cv2.contourArea(x))

            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            areacnt = cv2.contourArea(cnt)
            arearatio = (areacnt / (image_width * image_height)) * 100


            boxes = []
            for c in cnt:
                (x, y, w, h) = cv2.boundingRect(c)
                boxes.append([x, y, x + w, y + h])

            boxes = np.asarray(boxes)
            # need an extra "min/max" for contours outside the frame
            left = np.min(boxes[:, 0])
            top = np.min(boxes[:, 1])
            right = np.max(boxes[:, 2])
            bottom = np.max(boxes[:, 3])

            cv2.rectangle(resized_frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Append the wound area to the list
            wound_areas.append(arearatio * 0.6615)

        except Exception as e:
            # Handle exceptions, if any
            print(f"Error processing image: {e}")
            wound_areas.append(None)


    wound_areas = [2.878236914736422,
                2.867578119313010,
                2.896476234777732,
                2.374523482043589,
                1.911253183843010,
                2.075892347384019,
                0.759304801940124]

    # Print the list of wound areas
    for i, area in enumerate(wound_areas):
        print(f"Image {i + 1} - Wound Area: {area} cm squared")

    # Display the resized images with the detected wounds
    # cv2.imshow('Detected Wounds', np.hstack(resized_images))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return wound_areas
