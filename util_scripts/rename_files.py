import os

# Explain purpose of script for user in case someone runs it without understanding what it does
print("This script can be used to batch rename files to include their subdirectories as a prefix.")
print("They will then be moved into the target directory specified.")
print("This allows photos with originally overlapping names to be stored in a single directory.")

directory = input("Input Target Directory: ")

def rename_directory(path, root_path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path):
            print("Found directory", file_path)
            rename_directory(file_path, root_path)
        else:
            new_filename = file_path.removeprefix(root_path).replace("\\", "_")
            print(new_filename)
            os.rename(file_path, os.path.join(root_path, new_filename))

rename_directory(directory, directory)
