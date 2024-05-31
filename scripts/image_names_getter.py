import os
from openpyxl import Workbook

def list_files(root_directory):
    file_list = []
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_list.append(file_path)
    return file_list

def write_to_excel(file_list, excel_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "File List"

    sheet.append(["File Name"])
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        sheet.append([file_name])

    workbook.save(excel_path)


root_directory = "/Users/joudbabik/Desktop/Summer 2024/COMP 472/Datasets/Dirty /Focused"  # Change this to your root directory
excel_path = "/Users/joudbabik/Desktop/Summer 2024/image references - focused.xlsx"  # Output Excel file name

file_list = list_files(root_directory)
write_to_excel(file_list, excel_path)

print(f"File list has been written to {excel_path}")