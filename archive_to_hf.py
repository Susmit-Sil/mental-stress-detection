import os
import shutil
import zipfile
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN", "<YOUR_HF_TOKEN>")
REPO_NAME = "mental-stress-detection-dataset"
FOLDERS_TO_ARCHIVE = ["data", "datasets", "models", "Model-Evaluation", "results_auto"]

def zip_directory(folder_path, zip_path):
    print(f"Zipping {folder_path} to {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate relative path to store in zip
                rel_path = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, rel_path)
    print(f"Finished zipping {folder_path}. Size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")

def main():
    api = HfApi(token=TOKEN)
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    repo_id = f"{username}/{REPO_NAME}"
    print(f"Target Hugging Face Repository: {repo_id}")
    
    # Create dataset repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, token=TOKEN, repo_type="dataset", exist_ok=True)
        print("Dataset repository is ready.")
    except Exception as e:
        print(f"Error creating/checking repository: {e}")
        return

    uploaded_files = []
    
    # Archive and upload each folder
    for folder in FOLDERS_TO_ARCHIVE:
        if not os.path.exists(folder):
            print(f"Folder '{folder}' does not exist, skipping.")
            continue
            
        zip_name = f"{folder}.zip"
        zip_path = os.path.abspath(zip_name)
        
        try:
            zip_directory(folder, zip_path)
            
            print(f"Uploading {zip_name} to Hugging Face...")
            api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo=zip_name,
                repo_id=repo_id,
                repo_type="dataset",
                token=TOKEN
            )
            print(f"Successfully uploaded {zip_name}!")
            uploaded_files.append((folder, zip_path))
        except Exception as e:
            print(f"Failed to process/upload {folder}: {e}")
            # Clean up generated zip file on failure to prevent occupying space
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return

    # Delete local folders and zip files only after ALL uploads succeed
    print("\nAll uploads completed successfully! Cleaning up local space...")
    for folder, zip_path in uploaded_files:
        try:
            # Delete original folder
            print(f"Deleting local folder: {folder}")
            shutil.rmtree(folder)
            
            # Delete zip file
            print(f"Deleting temporary zip: {zip_path}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as e:
            print(f"Error during cleanup of {folder}: {e}")
            
    print("\nCleanup completed successfully!")

if __name__ == "__main__":
    main()
