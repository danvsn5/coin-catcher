# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import matplotlib
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

# —————————————————— Edge Detection with Horizontal and Vertical Scharr Kernals —————————————————— #
def edgeDetection(greyscale_pixel_array, image_width, image_height):
    edge_strength_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(1, image_height - 1):
        for col in range(1, image_width - 1):
            horizontal_strength = 3 * greyscale_pixel_array[row - 1][col - 1] + 10 * greyscale_pixel_array[row][col - 1] + 3 * greyscale_pixel_array[row + 1][col - 1] - 3 * greyscale_pixel_array[row - 1][col + 1] - 10 * greyscale_pixel_array[row][col + 1] - 3 * greyscale_pixel_array[row + 1][col + 1]
            vertical_strength = 3 * greyscale_pixel_array[row - 1][col - 1] + 10 * greyscale_pixel_array[row - 1][col] + 3 * greyscale_pixel_array[row - 1][col + 1] - 3 * greyscale_pixel_array[row + 1][col - 1] - 10 * greyscale_pixel_array[row + 1][col] - 3 * greyscale_pixel_array[row + 1][col + 1]
            
            horizontal_strength = horizontal_strength / 32
            vertical_strength = vertical_strength / 32            
            edge_strength = abs(horizontal_strength) + abs(vertical_strength)
            edge_strength = max(0, min(255, edge_strength))  # Ensure edge_strength is between 0 and 255
            edge_strength_array[row][col] = edge_strength
    
    return edge_strength_array

# ——————————————————————————————————— Blurring and Thresholding —————————————————————————————————— #
def meanFilter(greyscale_pixel_array, image_width, image_height):
    mean_filter_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(2, image_height - 2):
        for col in range(2, image_width - 2):
            mean_filter = 1/25 * (greyscale_pixel_array[row - 2][col - 2] + greyscale_pixel_array[row - 2][col - 1] + greyscale_pixel_array[row - 2][col] + greyscale_pixel_array[row - 2][col + 1] + greyscale_pixel_array[row - 2][col + 2] + greyscale_pixel_array[row - 1][col - 2] + greyscale_pixel_array[row - 1][col - 1] + greyscale_pixel_array[row - 1][col] + greyscale_pixel_array[row - 1][col + 1] + greyscale_pixel_array[row - 1][col + 2] + greyscale_pixel_array[row][col - 2] + greyscale_pixel_array[row][col - 1] + greyscale_pixel_array[row][col] + greyscale_pixel_array[row][col + 1] + greyscale_pixel_array[row][col + 2] + greyscale_pixel_array[row + 1][col - 2] + greyscale_pixel_array[row + 1][col - 1] + greyscale_pixel_array[row + 1][col] + greyscale_pixel_array[row + 1][col + 1] + greyscale_pixel_array[row + 1][col + 2] + greyscale_pixel_array[row + 2][col - 2] + greyscale_pixel_array[row + 2][col - 1] + greyscale_pixel_array[row + 2][col] + greyscale_pixel_array[row + 2][col + 1] + greyscale_pixel_array[row + 2][col + 2])
            mean_filter_array[row][col] = mean_filter
    
    return mean_filter_array

# perform thresholding on the input greyscale image to create a binary image
# the threshold value should be set to 22
def thresholding(greyscale_pixel_array, image_width, image_height):
    threshold_value = 22
    binary_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for row in range(image_height):
        for col in range(image_width):
            pixel_value = greyscale_pixel_array[row][col]
            if pixel_value < threshold_value:
                binary_pixel_array[row][col] = 0
            else:
                binary_pixel_array[row][col] = 255
    
    return binary_pixel_array

# ———————————————————————————————————— Dilations and Erosions ———————————————————————————————————— #
def erosionOnArray(binary_pixel_array, image_width, image_height):
    # implement with sliding window
    # create a new binary image with the eroded pixel values
    eroded_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    kernal = [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]

    # binary pixel array is the input binary image given in either 0 or 255
    for row in range(2, image_height - 2):
        for col in range(2, image_width - 2):
            eroded_pixel_array[row][col] = 255
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if kernal[i + 2][j + 2] == 1:
                        if binary_pixel_array[row + i][col + j] == 0:
                            eroded_pixel_array[row][col] = 0
                            break
                        
    return eroded_pixel_array

def dilationOnArray(binary_pixel_array, image_width, image_height):
    # implement with sliding window
    # create a new binary image with the dilated pixel values
    dilated_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    kernal = [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]

    # binary pixel array is the input binary image given in either 0 or 255
    for row in range(2, image_height - 2):
        for col in range(2, image_width - 2):
            dilated_pixel_array[row][col] = 0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if kernal[i + 2][j + 2] == 1:
                        if binary_pixel_array[row + i][col + j] == 255:
                            dilated_pixel_array[row][col] = 255
                            break
                        
    return dilated_pixel_array

# ———————————————————————————————————— Box Sizing and Grouping ——————————————————————————————————— #
def bfs(pixel_array, visited, row, col, image_width, image_height):
    queue = [(row, col)]
    visited[row][col] = True
    min_x = image_width
    min_y = image_height
    max_x = 0
    max_y = 0
    
    while queue:
        r, c = queue.pop(0)
        min_x = min(min_x, c)
        min_y = min(min_y, r)
        max_x = max(max_x, c)
        max_y = max(max_y, r)
        
        # Check neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr = r + dr
                nc = c + dc
                
                # Skip if out of bounds or already visited
                if nr < 0 or nr >= image_height or nc < 0 or nc >= image_width or visited[nr][nc]:
                    continue
                
                # Skip if not part of the connected component
                if pixel_array[nr][nc] != 255:
                    continue
                
                # Mark as visited and add to queue
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    return (min_x, min_y, max_x, max_y)


def connectedComponentAnalysis(binary_pixel_array, image_width, image_height):
    bounding_box_list = []
    visited = [[False] * image_width for _ in range(image_height)]
    
    for row in range(image_height):
        for col in range(image_width):
            if not visited[row][col] and binary_pixel_array[row][col] == 255:
                bounding_box = bfs(binary_pixel_array, visited, row, col, image_width, image_height)
                bounding_box_list.append(bounding_box)
    
    return bounding_box_list


# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_6'
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
    
    outputArray = edgeDetection(outputArray, image_width, image_height)
    
    outputArray = meanFilter(outputArray, image_width, image_height)
    outputArray = meanFilter(outputArray, image_width, image_height)
    outputArray = meanFilter(outputArray, image_width, image_height)
    
    outputArray = thresholding(outputArray, image_width, image_height)

    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)
    outputArray = dilationOnArray(outputArray, image_width, image_height)

    
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    outputArray = erosionOnArray(outputArray, image_width, image_height)
    
    bounding_box_list = connectedComponentAnalysis(outputArray, image_width, image_height)

    
        
    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    px_array = outputArray
    px_array = pyplot.imread(input_filename)

    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    coinCount = 0
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
                
        if(bbox_width > 0.95 * bbox_height and bbox_width < 1.05 * bbox_height):
            
            if(bbox_height > 100):
                coinCount = coinCount + 1
                rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
                axs.add_patch(rect)
                print(coinCount)
        
        
    if(coinCount == 1):
        matplotlib.pyplot.text(1, 1, "There is: " + str(coinCount) + " coin in this picture")
    else:
        matplotlib.pyplot.text(1, 1, "There are: " + str(coinCount) + " coins in this picture")

        
    
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
    