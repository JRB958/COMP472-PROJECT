# COMP 472 Artificial Intelligence  

## Team Name: OB_03

**Team Members:**
- Gevorg Alaverdyan (#40202177) - Data Specialist
- Jay Patel (#40203705) - Training Specialist
- Joud Babik (#40031039) - Evaluation Specialist

## Content Description

The `/scripts` folder contains scripts for data cleaning and visualization:

- **scripts/image_names_getter.py**: Retrieves file names and inputs them into an Excel file.
- **scripts/image_resize.py**: Removes any image smaller than 10KB and resizes images to 224x224 pixels.
- **scripts/aggregated_histograms.py**: Creates aggregated histograms for all images in a directory and its subdirectories.
- **scripts/Random_Histograms_Per_Folder.py**: Selects 15 random photos from a directory, generates histograms for them, and saves the histograms along with the images in a specified directory.

## Steps to Execute Code

### a) Data Cleaning Scripts

**image_names_getter.py**:
1. Copy and paste this script into your IDE of choice.
2. Update the variables:
   - `root_directory = "/Users/username/..."` (Set to your root directory)
   - `excel_path = "/Users/username/.../image_references_focused.xlsx"` (Set to your desired output Excel file name)
3. Run the script.

**image_resize.py**:
1. Copy and paste this script into your IDE of choice.
2. Update the variables:
   - `file_path = "C:\\Users\\gevor\\Downloads\\colored2"` (Set to your input directory. Ensure this directory exists before running the script.)
   - `outfile = "C:\\Users\\gevor\\Downloads\\fixed\\"` (Set to your desired output directory. Ensure this directory exists before running the script.)
3. Run the script.

### b) Data Visualization Scripts

**aggregated_histograms.py**:
1. Copy and paste this script into your IDE of choice.
2. Run the script.
3. When prompted, paste the absolute path of the directory you want to aggregate pixel intensity for.

**Random_Histograms_Per_Folder.py**:
1. Copy and paste this script into your IDE of choice.
2. Update the paths:
   - Set the relative path to your source directory on line 207.
   - Set the relative path to your destination directory on line 209.
3. Run the script.
