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
import pathspec
import re


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
    ignore_patterns = pathspec.PathSpec.from_lines('gitwildmatch', Path('.gitignore').read_text().splitlines())
    all_files = [f for f in Path('.').rglob('*')]
    ignored = [str(f) for f in all_files if ignore_patterns.match_file(str(f))]
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

def get_current_version():
    """Get the current version from TODO file."""
    try:
        with open('TODO', 'r') as f:
            content = f.read()
            # Find the first version header
            match = re.search(r'## (v\d+\.\d+) -', content)
            if match:
                return match.group(1)
        return "v1.0"  # Default if not found
    except Exception as e:
        logging.warning(f"Could not read version from TODO: {e}")
        return "v1.0"

def get_todo_completion_percentage():
    """Calculate how much of the current version's TODO is completed."""
    try:
        with open('TODO', 'r') as f:
            content = f.read()

        # Find current version section
        current_version = get_current_version()
        pattern = rf'## {re.escape(current_version)} -.*?(?=## v|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return 0.0

        section = match.group(0)
        total_items = len(re.findall(r'^- ', section))
        completed_items = len(re.findall(r'^- \[x\]', section))

        if total_items == 0:
            return 100.0
        return (completed_items / total_items) * 100.0
    except Exception as e:
        logging.warning(f"Could not calculate TODO completion: {e}")
        return 0.0

def create_git_tag(version, build_type):
    """Create a git tag for release builds."""
    if build_type.lower() != 'release':
        logging.info("Skipping tag creation for non-release build")
        return

    try:
        tag_name = f"{version}-release"
        logging.info(f"Creating git tag: {tag_name}")

        # Check if tag already exists
        result = subprocess.run(['git', 'tag', '-l', tag_name],
                              capture_output=True, text=True)
        if tag_name in result.stdout.strip():
            logging.info(f"Tag {tag_name} already exists, skipping")
            return

        # Create annotated tag
        subprocess.run(['git', 'tag', '-a', tag_name, '-m', f'Release {version}'],
                      check=True)
        logging.info(f"Successfully created tag: {tag_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create git tag: {e}")
        raise

def create_next_version_branch(current_version):
    """Create a new branch for the next version if current version TODO is complete."""
    try:
        completion = get_todo_completion_percentage()
        if completion < 100.0:
            logging.info(f"Current version {current_version} not complete ({completion:.1f}%), skipping branch creation")
            return

        # Parse version number
        match = re.match(r'v(\d+)\.(\d+)', current_version)
        if not match:
            logging.warning(f"Could not parse version {current_version}")
            return

        major, minor = int(match.group(1)), int(match.group(2))
        next_version = f"v{major}.{minor + 1}"

        branch_name = f"development/{next_version}"

        # Check if branch already exists
        result = subprocess.run(['git', 'branch', '-l', branch_name],
                              capture_output=True, text=True)
        if branch_name in result.stdout.strip():
            logging.info(f"Branch {branch_name} already exists")
            return

        # Create and checkout new branch
        logging.info(f"Creating new branch for next version: {branch_name}")
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)

        # Update TODO file for new version
        update_todo_for_new_version(next_version)

        logging.info(f"Successfully created and switched to branch: {branch_name}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to create next version branch: {e}")
        raise

def update_todo_for_new_version(new_version):
    """Update TODO file to start new version section."""
    try:
        with open('TODO', 'r') as f:
            content = f.read()

        # Remove completed version sections
        lines = content.split('\n')
        new_lines = []
        skip_section = False

        for line in lines:
            if line.startswith('## ') and ' - ' in line:
                version_match = re.match(r'## (v\d+\.\d+) -', line)
                if version_match:
                    current_completion = get_todo_completion_percentage()
                    if current_completion >= 100.0:
                        skip_section = True
                        continue
                    else:
                        skip_section = False

            if not skip_section or not line.strip().startswith('- '):
                new_lines.append(line)

        # Add new version header
        new_lines.insert(0, f"## {new_version} - Next Version Features")
        new_lines.insert(1, "- [ ] Feature placeholder")
        new_lines.insert(2, "")

        with open('TODO', 'w') as f:
            f.write('\n'.join(new_lines))

        logging.info(f"Updated TODO file for new version: {new_version}")

    except Exception as e:
        logging.error(f"Failed to update TODO file: {e}")

def perform_git_operations(branch, remotes, build_type):
    """Perform git add, commit, and push operations with version management."""
    try:
        current_version = get_current_version()
        completion = get_todo_completion_percentage()

        logging.info(f"Current version: {current_version}, Completion: {completion:.1f}%")

        # Create tag for release builds
        create_git_tag(current_version, build_type)

        # Create next version branch if current is complete
        create_next_version_branch(current_version)

        # Skip git operations if there are no changes or if we're in a problematic state
        logging.info("Skipping git operations to avoid build failures")
        #Comment out git operations to prevent build failures
        logging.info("Performing git add .")
        subprocess.run(['git', 'add', '.'], check=True)

        logging.info("Performing git commit")
        commit_msg = f'Build {current_version} - {completion:.1f}% complete'
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

        for remote in remotes:
            logging.info(f"Pushing to remote: {remote}")
            subprocess.run(['git', 'push', remote, branch], check=True)

        logging.info("Git operations completed")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git operation failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Post-build automation script')
    parser.add_argument('--source-dir', required=True, help='Source directory to copy')
    parser.add_argument('--build-dir', required=True, help='Build directory (for reference)')
    parser.add_argument('--target-executable', required=True, help='Target executable file path')
    parser.add_argument('--build-type', default='Release', help='Build type (Release/Debug)')

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    build_dir = Path(args.build_dir)
    target_executable = Path(args.target_executable)
    build_type = args.build_type

    # Validate inputs
    if not source_dir.exists() or not source_dir.is_dir():
        logging.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    if not target_executable.exists():
        logging.error(f"Target executable does not exist: {target_executable}")
        sys.exit(1)

    destinations = [
        Path('/mnt/hdd-archive/code/CoreAI3D'),
        Path('/mnt/ext_hdd/code/CoreAI3D')
    ]

    try:
        # Copy to destinations
        for dest in destinations:
            delete_directory(dest)
            copy_source_to_destination(source_dir, dest)

        # Perform git operations
        branch = get_current_branch()
        remotes = get_git_remotes()
        perform_git_operations(branch, remotes, build_type)

        logging.info("Post-build automation completed successfully")

    except Exception as e:
        logging.error(f"Post-build automation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()