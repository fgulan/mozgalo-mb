"""
Utility Functions
"""
import os


def list_dir(dir_path, extension=None):
    """
    Creates a list of directory files. It is not recursive and it lists FILES ONLY.
    If the extension argument is given it will filter files that end with the given extension.

    :param extension: File extension filter
    :param dir_path: Directory path
    :return: List of absolute file paths in the given directory
    """
    files = [os.path.join(dir_path, p) for p in os.listdir(dir_path) if
                os.path.isfile(os.path.join(dir_path, p))]
    if extension:
        return list(filter(lambda x: x.endswith(extension), files))
    else:
        return files

print(list_dir("/home/bartol/Documents"))




