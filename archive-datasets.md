# Plan: Archive All Heavy Folders to Hugging Face & Push Code to GitHub (Workspace Copy)

This plan details the steps to archive all heavy folders to Hugging Face, freeze dependency versions, update `.gitignore`, and push remaining scripts to GitHub.

## Project Type
- BACKEND/DATA PIPELINE (Python)

## Success Criteria
- [x] Dependencies locked in `requirements.txt`
- [x] Zipped directories (`data`, `datasets`, `models`, `Model-Evaluation`, `results_auto`) uploaded to `mental-stress-detection-dataset` on Hugging Face
- [x] Local directories deleted successfully after upload
- [x] `.gitignore` updated to ignore large folders and zip files
- [x] Remaining code pushed to GitHub
- [x] Documentation files (`clone_walkthrough.txt` and `AGENT_SETUP.md`) created

## Tech Stack
- Python 3.x
- `huggingface_hub` Python SDK
- Git

## File Structure Changes
- [NEW] `archive_to_hf.py` (Script to zip, upload, and clean up)
- [MODIFY] `requirements.txt` (Locked dependency versions)
- [MODIFY] `.gitignore` (Ignore large folders and zip files)
- [NEW] `clone_walkthrough.txt` (Recovery instructions)
- [NEW] `AGENT_SETUP.md` (Agent instructions)

## Task Breakdown

### Task 1: Lock Dependencies and Update .gitignore
- **Agent**: `devops-engineer`
- **Priority**: P0
- **Input**: Workspace environment
- **Output**: `requirements.txt` and `.gitignore`
- **Verify**: Files contain correct entries.

### Task 2: Upload Script Update & Execution
- **Agent**: `backend-specialist`
- **Priority**: P0
- **Input**: `archive_to_hf.py` configuration
- **Output**: Uploaded files on Hugging Face and local directories deleted.
- **Verify**: Large folders are successfully removed from local disk.

### Task 3: Push to GitHub
- **Agent**: `orchestrator`
- **Priority**: P0
- **Input**: Clean codebase
- **Output**: Pushed commit to remote repo.
- **Verify**: `git status` shows up-to-date branch.

### Task 4: Documentation Creation
- **Agent**: `documentation-writer`
- **Priority**: P1
- **Input**: Completion status
- **Output**: `clone_walkthrough.txt` and `AGENT_SETUP.md`
- **Verify**: Both files are present and contain clear details.

## ✅ PHASE X COMPLETE
- Lint: ✅ Pass
- Security: ✅ Secrets removed (passed GitHub push protection)
- Build: ✅ Success
- Date: 2026-08-05
