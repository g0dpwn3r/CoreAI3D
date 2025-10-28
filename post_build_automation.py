#!/usr/bin/env python3
"""
Post-build automation script for CoreAI3D project.
Handles copying source code to backup locations and performing git operations.
"""

import argparse
import shutil
import subprocess
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('post_build_automation.log')
    ]
)

def ignore_function(src, names):
    """
    Function to ignore certain files/directories during copy.
    Excludes build artifacts, version control, and temporary files.
    """
    ignore_patterns = {
        '.git',
        '__pycache__',
        'CMakeFiles',
        'build',
        'x64',
        'Debug',
        'Release',
        '*.log',
        '*.tlog',
        '*.exe.recipe',
        '*.vcxproj.FileListAbsolute.txt',
        'unsuccessfulbuild',
        'vcpkg'  # Exclude vcpkg directory to avoid infinite symlink issues
    }

    ignored = []
    for name in names:
        if name in ignore_patterns or name.startswith('.') or name.endswith('.log'):
            ignored.append(name)
    return ignored

def delete_directory(path):
    """Safely delete a directory tree."""
    try:
        if path.exists():
            logging.info(f"Deleting directory: {path}")
            shutil.rmtree(path)
        else:
            logging.info(f"Directory does not exist, skipping deletion: {path}")
    except Exception as e:
        logging.error(f"Failed to delete {path}: {e}")
        raise

def copy_source_to_destination(src, dest):
    """Copy source directory to destination, excluding build artifacts."""
    try:
        logging.info(f"Copying {src} to {dest}")
        shutil.copytree(src, dest, ignore=ignore_function)
        logging.info(f"Successfully copied to {dest}")
    except Exception as e:
        logging.error(f"Failed to copy {src} to {dest}: {e}")
        raise

def get_current_branch():
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current branch: {e}")
        raise

def get_git_remotes():
    """Get list of git remotes."""
    try:
        result = subprocess.run(
            ['git', 'remote'],
            capture_output=True,
            text=True,
            check=True
        )
        remotes = [remote.strip() for remote in result.stdout.split('\n') if remote.strip()]
        return remotes
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get git remotes: {e}")
        raise

def setup_git_lfs():
    """Set up Git LFS for large files."""
    try:
        # Check if git-lfs is available
        subprocess.run(['git', 'lfs', 'version'], capture_output=True, check=True)
        logging.info("Git LFS is available")

        # Install Git LFS hooks
        subprocess.run(['git', 'lfs', 'install'], check=True)
        logging.info("Git LFS installed")

        # Track large files with LFS
        large_file_patterns = [
            "*.log",
            "*.tlog",
            "*.exe.recipe",
            "*.vcxproj.FileListAbsolute.txt",
            "assets/*",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.bmp",
            "*.ico",
            "*.pdf",
            "*.docx",
            "*.xlsx",
            "*.ipynb"
        ]

        for pattern in large_file_patterns:
            try:
                subprocess.run(['git', 'lfs', 'track', pattern], check=True)
                logging.info(f"Tracking {pattern} with Git LFS")
            except subprocess.CalledProcessError:
                logging.warning(f"Failed to track {pattern} with Git LFS")

        # Add .gitattributes if it exists
        if Path('.gitattributes').exists():
            subprocess.run(['git', 'add', '.gitattributes'], check=True)
            logging.info("Added .gitattributes to staging")

    except subprocess.CalledProcessError:
        logging.warning("Git LFS is not available or failed to set up. Continuing without LFS.")

def perform_git_operations(branch, remotes):
    """Perform git add, commit, and push operations."""
    try:
        # Set up Git LFS first
        setup_git_lfs()

        # Check if there are any changes to commit
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            logging.info("No changes to commit, skipping git operations")
            return

        logging.info("Performing git add -A")
        subprocess.run(['git', 'add', '-A'], check=True)

        logging.info("Performing git commit")
        subprocess.run(['git', 'commit', '-m', 'Another successful build with Git LFS support'], check=True)

        # Ensure we push to GitHub, GitLab, and Bitbucket if configured
        target_remotes = ['origin', 'github', 'gitlab', 'bitbucket']
        available_remotes = [r for r in remotes if r in target_remotes or any(tr in r for tr in target_remotes)]

        if not available_remotes:
            available_remotes = remotes  # Fallback to all remotes

        for remote in available_remotes:
            try:
                logging.info(f"Pushing to remote: {remote}")
                result = subprocess.run(['git', 'push', remote, branch], capture_output=True, text=True, check=True)
                logging.info(f"Successfully pushed to {remote}")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Failed to push to remote {remote}: {e}")
                if e.stdout:
                    logging.warning(f"Push stdout: {e.stdout}")
                if e.stderr:
                    logging.warning(f"Push stderr: {e.stderr}")
                logging.warning(f"Continuing with other operations despite push failure to {remote}")

        logging.info("Git operations completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {e}")
        # Don't raise exception to avoid failing the build
        logging.warning("Git operations failed, but build will continue")

def main():
    parser = argparse.ArgumentParser(description='Post-build automation script')
    parser.add_argument('--source-dir', required=True, help='Source directory to copy')
    parser.add_argument('--build-dir', required=True, help='Build directory (for reference)')
    parser.add_argument('--target-executable', required=True, help='Target executable file path')

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    build_dir = Path(args.build_dir)
    target_executable = Path(args.target_executable)

    # Validate inputs
    if not source_dir.exists() or not source_dir.is_dir():
        logging.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    if not target_executable.exists():
        logging.error(f"Target executable does not exist: {target_executable}")
        sys.exit(1)

    destinations = [
        Path('/mnt/hdd-archive/code/CoreAI3D'),
        # Path('/mnt/ext-hdd/code/CoreAI3D')  # Commented out due to permission issues
    ]

    try:
        # Copy to destinations
        for dest in destinations:
            delete_directory(dest)
            copy_source_to_destination(source_dir, dest)

        # Perform git operations
        branch = get_current_branch()
        remotes = get_git_remotes()
        perform_git_operations(branch, remotes)

        logging.info("Post-build automation completed successfully")

    except Exception as e:
        logging.error(f"Post-build automation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()