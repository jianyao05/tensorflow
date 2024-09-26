import os

data_dir = r'/exercise_data'

# Check if the directory exists
if not os.path.exists(data_dir):
    raise Exception(f"The directory {data_dir} does not exist. Please check the path.")
else:
    print(f"The directory {data_dir} was found.")
    # List files and directories in the path
    print("Contents of the directory:")
    for item in os.listdir(data_dir):
        print(item)
