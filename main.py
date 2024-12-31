import cv2
import numpy as np

#ds


def print_menu():
    print("\nImage Processing Menu:")
    print("1. Read Image")
    print("2. Convert to Grayscale")
    print("3. Print Image Info")
    print("4. Reduce Number of Levels")
    print("5. Replace Corresponding Pixels by Their Average")
    print("6. Blur the Grayscale Image")
    print("7. Apply Median Filter")
    print("8. Represent with Alpha Channel")
    print("9. Exit")

# Function placeholders (to be implemented)
def read_image():
    img = cv2.imread('img_1.png')
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img  # Return the image

def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def print_image_info(img):
    sizeOfTheImage = img.shape
    print(f"Image size (height, width): {sizeOfTheImage}")

    # Get the unique gray levels
    graylevel = np.unique(img)
    print(f"Number of gray levels: {len(graylevel)}")

    # Find the min and max gray levels
    min_gray = np.min( graylevel)
    max_gray = np.max(graylevel)

    print(f"Min gray level: {min_gray}")
    print(f"Max gray level: {max_gray}")
    print(f"Total number of pixels: {img.size}")

def reduce_levels(img, r):
    scaleFactor = 255 //( pow(2, r))
    reduced_img = (img // scaleFactor) * scaleFactor  # Corrected processing
    cv2.namedWindow('Reduced gray Levels', cv2.WINDOW_NORMAL)
    cv2.imshow('Reduced gray Levels' , reduced_img)  # Display the modified image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return reduced_img

def replace_pixels_by_average(img,r):
    image = img

    # Get the dimensions of the image
    h, w = image.shape

    # Ensure that the block size r is odd
    if r % 2 == 0:
        raise ValueError("r must be an odd number")

    # Create an empty array for the new image (same size as original)
    new_image = np.zeros_like(image)

    # Iterate over the image in non-overlapping blocks of size r x r
    for i in range(0, h, r):
        for j in range(0, w, r):
            # Get the block (handling boundary conditions if the block is smaller at the edges)
            block = image[i:i + r, j:j + r]

            # Compute the average value of the block
            avg = np.mean(block, dtype=int)

            # Assign the average value to all pixels in the block
            new_image[i:i + r, j:j + r] = avg

    # Return the processed image
    cv2.namedWindow('pixel average  Image', cv2.WINDOW_NORMAL)
    cv2.imshow('pixel average  Image', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#12240981
def blur_image(img):
    for _ in range(9):
        bluredImage = cv2.blur(img, (8, 1))
    cv2.namedWindow('Blured Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Blured Image', bluredImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def apply_median_filter(img):
    filtered_image = cv2.medianBlur(image, 5)
    cv2.namedWindow('Median Filter', cv2.WINDOW_NORMAL)
    cv2.imshow('Median Filter', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def represent_with_alpha(img):
    height, width = image.shape

    # Quantize the image to 26 levels (for the letters A-Z)
    quantized_image = (image // 10).astype(np.uint8)  # 256 levels reduced to 26 levels

    # Define the English alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Create an empty list to store the result as strings
    text_representation = []

    # Map each quantized level to a letter
    for y in range(height):
        row = ""
        for x in range(width):
            # Map the quantized value to a letter
            letter = alphabet[quantized_image[y, x]]
            row += letter
        text_representation.append(row)

    return text_representation

    # Load the grayscale image


image = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)

# Quantize the image and map to the alphabet


def main():
    imge = None  # Initialize the image variable
    gray = None
    while True:
        print_menu()
        choice = input("Enter your choice: ")

        if choice == '1':

            imge = read_image()

        elif choice == '2':
            if imge is not None:
                gray = convert_to_grayscale(imge)
                cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
                cv2.imshow('Grayscale', gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Please read an image first.")

        elif choice == '3':
            if gray is not None:  # Check if the image is read
                print_image_info(gray)
            else:
                print("Please read an image first.")

        elif choice == '4':
            if gray is not None:
                try:
                    r = int(input("Enter the number of levels to reduce (1-8): "))
                    gray = reduce_levels(gray, r)
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            else:
                print("Please read an image first.")

        elif choice == '5':
            while True:
                try:
                    r = int(input("Enter the size of the filter: "))
                    if r % 2 == 0:
                        print("Please enter an odd number for the filter size.")
                        continue
                    replace_pixels_by_average(gray, r)
                    break
                except ValueError:
                    print("Invalid input, please enter a valid integer.")
        elif choice == '6':
            if gray is not None:
                blur_image(gray)
            else:
                print("Please read an image first.")

        elif choice == '7':
            if gray is not None:
                apply_median_filter(gray)
            else:
                print("Please read an image first.")


        elif choice == '8':
            if gray is not None:
                text_image = represent_with_alpha(gray)

                # Save the result as a text file
                try:
                    with open("image_as_text.txt", "w") as f:
                        for line in text_image:
                            f.write(line + "\n")
                    print(
                        "The image has been successfully represented using alphabet letters and saved to 'image_as_text.txt'.")
                except IOError as e:
                    print(f"An error occurred while writing to the file: {e}")

        elif choice == '9' :
             print("Thank you!")
             break


        else:
            print("Please read an image first.")

if __name__ == "__main__":
    main()
