import os

def generate_directory_structure_excluding_specific_files(assets_dir="./", output_file_name="directory_structure.txt"):
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the Assets directory exists
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Full path for the output file
    output_file_path = os.path.join(assets_dir, output_file_name)
    
    def write_structure(dir_path, prefix=""):
        entries = sorted(os.listdir(dir_path))
        entries_count = len(entries)
        
        for i, entry in enumerate(entries):
            entry_path = os.path.join(dir_path, entry)
            
            if entry in [".vs", 
            "env", 
            "__pycache__", 
            "Dataset", 
            "data", 
            "dt_str.py", 
            ".git", 
            "PU-GAN"]:
                continue
            
            is_last = (i == entries_count - 1)
            connector = "└── " if is_last else "├── "
            
            # Write the current entry
            f.write(f"{prefix}{connector}{entry}\n")
            
            # If the entry is a directory, recursively process its contents
            if os.path.isdir(entry_path):
                new_prefix = prefix + ("    " if is_last else "│   ")
                write_structure(entry_path, new_prefix)
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"{os.path.basename(current_directory)}/\n")
        write_structure(current_directory)

    print(f"Directory structure saved to: {output_file_path}")

# Run the function to generate the directory structure
generate_directory_structure_excluding_specific_files()