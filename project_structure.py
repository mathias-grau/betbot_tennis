import os

def print_project_structure(base_path):
    # List of directories and files to ignore
    ignore_list = ['.git', '__pycache__']  # Add more directories or file names as needed
    
    # Print the current directory structure
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in ignore_list]
        files[:] = [f for f in files if f not in ignore_list]
        
        level = root.replace(base_path, '').count(os.sep)
        indent = '    ' * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        
        subindent = '    ' * (level + 1)
        for file in files:
            print('{}{}'.format(subindent, file))

# Example usage:
if __name__ == "__main__":
    project_path = "."  # Replace with your actual project path
    print_project_structure(project_path)
