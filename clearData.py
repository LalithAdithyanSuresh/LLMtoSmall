import os
import shutil

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    print(f"Clearing folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"  Deleted file: {filename}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"  Deleted directory: {filename}")
        except Exception as e:
            print(f"  Failed to delete {filename}. Reason: {e}")

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define targets relative to the script directory
    folders_to_clear = [
        os.path.join(script_dir, "dataPool"),
        os.path.join(script_dir, "tasks")
    ]
    
    for folder in folders_to_clear:
        clear_folder(folder)
