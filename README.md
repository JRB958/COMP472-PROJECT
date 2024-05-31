# COMP 472 Artificial Intelligence  

## Team Name: OB_03

<pre>
Team Members:
Gevorg Alaverdyan (#40202177) - Data Specialist
Jay Patel (#40203705) - Training Specialist
Joud Babik (#40031039) - Evaluation Specialist
</pre>

### Content Description
/scripts folder contains all the scripts that helped us with data cleaning and labelling

scripts/image_names_getter.py: will get you file names and input in an excel

scripts/image_resize.py: will remove any image lower than 10KB and resize to 224x224px image.

### Steps to execute code
a) Data Cleaning
scripts/image_names_getter.py: 
- Copy paste this script in your IDE of choice 

- root_directory = "/Users/username/..."  # Change this to your root directory

- excel_path = ""/Users/username/.../image references - focused.xlsx"  # Change Output Excel file name 

- Run the script

scripts/image_resize.py: 
- Copy paste this script in your IDE of choice 

- file_path = "C:\\Users\\gevor\\Downloads\\colored2" change the input directory to where you store the pictures. The directory should exist before running the script.

- outfile = "C:\\Users\\gevor\\Downloads\\fixed\\" change this string to your desired output directory. The directory should exist before running the script.

- Run the script

b)