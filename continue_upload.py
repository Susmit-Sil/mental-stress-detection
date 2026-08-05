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
    print(f"Target repo: {repo_id}")

    try:
        create_repo(repo_id=repo_id, token=TOKEN, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo check failed: {e}")
        return

    uploaded_files = []

    for folder in FOLDERS_TO_ARCHIVE:
        zip_name = f"{folder}.zip"
        zip_path = os.path.abspath(zip_name)

        # Check if the folder or the pre-existing zip file exists
        if not os.path.exists(folder) and not os.path.exists(zip_path):
            print(f"Neither folder '{folder}' nor zip '{zip_name}' exists. Skipping.")
            continue

        # If zip doesn't exist, create it from the folder
        if not os.path.exists(zip_path):
            try:
                zip_directory(folder, zip_path)
            except Exception as e:
                print(f"Failed to zip {folder}: {e}")
                continue
        else:
            print(f"Using pre-existing zip: {zip_name} ({os.path.getsize(zip_path) / 1024 / 1024:.2f} MB)")

        # Upload
        print(f"Uploading {zip_name}...")
        try:
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
            print(f"Upload failed for {zip_name}: {e}")
            return

    # Clean up
    print("\nAll uploads completed successfully! Cleaning up local space...")
    for folder, zip_path in uploaded_files:
        try:
            if os.path.exists(folder):
                print(f"Deleting local folder: {folder}")
                shutil.rmtree(folder)
            if os.path.exists(zip_path):
                print(f"Deleting temporary zip: {zip_path}")
                os.remove(zip_path)
        except Exception as e:
            print(f"Error cleaning up {folder}: {e}")

    print("\nWorkspace cleanup complete!")

if __name__ == "__main__":
    main()
