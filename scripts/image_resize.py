import os

from PIL import Image

# Change to  your input folder
file_path = "C:\\Users\\gevor\\Downloads\\colored2"
directory = os.fsencode(file_path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_size = os.path.getsize(os.path.join(file_path, filename))

    # This checks for files higher than 10KB
    if file_size >= 10000:
        print(filename)
        infile = os.path.join(file_path, filename)
        #change to your output folder
        outfile = "C:\\Users\\gevor\\Downloads\\fixed\\"+filename
        try:
            with Image.open(infile) as im:
                # Print the original size
                print(f"Original size: {im.size}")

                # Define the target size
                target_size = (224, 224)

                # Resize the image while maintaining aspect ratio
                im.thumbnail(target_size, Image.Resampling.LANCZOS)

                # Create a new image with the target size and white background
                new_im = Image.new("RGB", target_size, (255, 255, 255))

                # Calculate the position to paste the resized image onto the new image
                left = (target_size[0] - im.size[0]) // 2
                top = (target_size[1] - im.size[1]) // 2
                new_im.paste(im, (left, top))

                # Save the resized and padded image
                new_im.save(outfile, "JPEG")

                # Output the new size of the resized image
                print(f"Resized size: {new_im.size}")
        except IOError:
            print(f"Cannot create thumbnail for '{infile}'")
        continue
    else:
        os.remove(os.path.join(file_path, filename))
        continue
