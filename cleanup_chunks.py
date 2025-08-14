import os
import shutil
import argparse
from pathlib import Path


def cleanup_chunk_directories(base_path, chunk_length, overlap, dry_run=True):
    """
    Remove chunk directories with specific length and overlap parameters.
    
    Args:
        base_path (str): Base path containing patient folders
        chunk_length (float): Chunk length to match
        overlap (float): Overlap to match
        dry_run (bool): If True, only print what would be deleted without actually deleting
    """
    # Format the directory name to match what was created
    target_dir_name = f"{chunk_length:.1f}s_{overlap:.1f}s_overlap"
    
    print(f"Looking for directories named: {target_dir_name}")
    print(f"Base path: {base_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)
    
    deleted_count = 0
    total_size = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == target_dir_name:
                full_path = os.path.join(root, dir_name)
                
                # Calculate directory size
                try:
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(full_path)
                        for filename in filenames
                    )
                    total_size += dir_size
                    
                    if dry_run:
                        print(f"Would delete: {full_path} (Size: {dir_size / (1024*1024):.2f} MB)")
                    else:
                        shutil.rmtree(full_path)
                        print(f"Deleted: {full_path} (Size: {dir_size / (1024*1024):.2f} MB)")
                        deleted_count += 1
                        
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
    
    print("-" * 60)
    if dry_run:
        print(f"DRY RUN: Would delete {deleted_count} directories")
        print(f"Total size that would be freed: {total_size / (1024*1024):.2f} MB")
    else:
        print(f"Successfully deleted {deleted_count} directories")
        print(f"Total space freed: {total_size / (1024*1024):.2f} MB")


def list_all_chunk_directories(base_path):
    """
    List all chunk directories found in the base path.
    
    Args:
        base_path (str): Base path containing patient folders
    """
    print(f"Scanning for chunk directories in: {base_path}")
    print("-" * 60)
    
    chunk_dirs = set()
    
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if "s_overlap" in dir_name:
                chunk_dirs.add(dir_name)
    
    if chunk_dirs:
        print("Found chunk directories:")
        for dir_name in sorted(chunk_dirs):
            print(f"  - {dir_name}")
    else:
        print("No chunk directories found.")
    
    print(f"\nTotal unique chunk directory types: {len(chunk_dirs)}")


def main():
    # Configuration parameters - change these values as needed
    CHUNK_LENGTH = 3  # seconds
    OVERLAP = 1       # seconds
    BASE_PATH = "data/created_data"  # Base path containing patient folders
    DRY_RUN = True      # Set to False to actually delete directories
    
    print(f"Cleanup Configuration:")
    print(f"  Chunk length: {CHUNK_LENGTH}s")
    print(f"  Overlap: {OVERLAP}s")
    print(f"  Base path: {BASE_PATH}")
    print(f"  Dry run: {DRY_RUN}")
    print("-" * 60)
    
    if DRY_RUN:
        print("DRY RUN MODE - No files will be deleted")
        print("To actually delete files, change DRY_RUN = False in the script")
        print("-" * 60)
    
    # Confirm deletion if not in dry-run mode
    if not DRY_RUN:
        confirm = input(f"Are you sure you want to delete all '{CHUNK_LENGTH:.1f}s_{OVERLAP:.1f}s_overlap' directories? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    cleanup_chunk_directories(
        BASE_PATH, 
        CHUNK_LENGTH, 
        OVERLAP, 
        dry_run=DRY_RUN
    )


if __name__ == "__main__":
    main()