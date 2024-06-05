# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

# ——————————————————————————————————————————————— - —————————————————————————————————————————————— #
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(image_height):
        for col in range(image_width):
            r = pixel_array_r[row][col]
            g = pixel_array_g[row][col]
            b = pixel_array_b[row][col]
            
            grey = round(0.3 * r + 0.6 * g + 0.1 * b)
            
            greyscale_pixel_array[row][col] = grey
    
    return greyscale_pixel_array


# ——————————————————————————————————————————————— - —————————————————————————————————————————————— #

def computeMinMax(greyscale_pixel_array, image_width, image_height):
    min_value = 255
    max_value = 0
    
    for row in range(image_height):
        for col in range(image_width):
            pixel_value = greyscale_pixel_array[row][col]
            if pixel_value < min_value:
                min_value = pixel_value
            if pixel_value > max_value:
                max_value = pixel_value

    return min_value, max_value
    
# —————————————————————————————————————— Contrast Stretching ————————————————————————————————————— #

def computeCumulativeHistogram(pixel_array, nr_bins):
    histogram = [0] * nr_bins
    
    for row in pixel_array:
        for pixel in row:
            histogram[pixel] += 1
    
    cumulative_histogram = [0] * nr_bins
    cumulative_sum = 0
    for i in range(nr_bins):
        cumulative_sum += histogram[i]
        cumulative_histogram[i] = cumulative_sum
    
    return cumulative_histogram

def contrastStretching(greyscale_pixel_array, image_width, image_height):
   
    # compute the cumulative histogram for the input greyscale image
    nr_bins = 256
    cumulative_histogram = computeCumulativeHistogram(greyscale_pixel_array, nr_bins)
    
    # determine the 5th and 95th percentile pixel values
    total_pixels = image_width * image_height
    lower_percentile = 0.05
    upper_percentile = 0.95
    
    lower_pixel_value = 0
    upper_pixel_value = 255
    
    lower_threshold = lower_percentile * total_pixels
    upper_threshold = upper_percentile * total_pixels
    
    for i in range(nr_bins):
        if cumulative_histogram[i] >= lower_threshold:
            lower_pixel_value = i
            break
    
    for i in range(nr_bins - 1, -1, -1):
        if cumulative_histogram[i] <= upper_threshold:
            upper_pixel_value = i
            break
    
    print("lower pixel value: ", lower_pixel_value)
    print("upper pixel value: ", upper_pixel_value)
    
    # create a new greyscale image with the stretched pixel values
    stretched_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(image_height):
        for col in range(image_width):
            pixel_value = greyscale_pixel_array[row][col]
            if pixel_value < lower_pixel_value:
                pixel_value = 0
            elif pixel_value > upper_pixel_value:
                pixel_value = 255
            else:
                pixel_value = math.floor((pixel_value - lower_pixel_value) * 255 / (upper_pixel_value - lower_pixel_value))
            
            stretched_pixel_array[row][col] = pixel_value
    
    return stretched_pixel_array






# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_1'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################
    
    # Create a greyscale image using the provided channel ratios and rounding
    iniGreyscaleArray = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
               
               
    outputArray = contrastStretching(iniGreyscaleArray, image_width, image_height)
    
        
    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    bounding_box_list = [[150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    px_array = outputArray
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        
    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    