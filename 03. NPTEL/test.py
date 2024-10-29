import os

# Define the base directory where the folders will be created
base_dir = "D:/DEEP LEARNING/03. NPTEL/"  # Change this to the path where you want the folders to be created

# Loop to create folders from Week01 to Week12
for i in range(1, 13):
    # Format folder name as Week01, Week02, ..., Week12
    folder_name = f"Week{i:02}"
    
    # Define the folder path
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if the folder already exists
    if os.path.exists(folder_path):
        print(f"{folder_name} already exists, skipping...")
        continue  # Skip this folder if it exists
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the NOTES.md file path within the folder
    notes_file_path = os.path.join(folder_path, "NOTES.md")
    
    # Create the NOTES.md file
    with open(notes_file_path, "w") as f:
        f.write(f"# Notes for {folder_name}\n\n")  # Write a header to the NOTES.md file

print("Folders and NOTES.md files created successfully.")
