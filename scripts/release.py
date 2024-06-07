"""@private"""

import subprocess
import sys
import argparse
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="🤖 Release SDK - %(message)s")


def run_command(command, check=True):
    logging.info(f"Running command: {command}")
    result = subprocess.run(
        command, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def check_git_status():
    # Check if there are uncommitted changes
    logging.info("Checking for uncommitted changes...")
    status_output = run_command("git status --porcelain", check=False)
    if status_output:
        logging.error(
            "Your working directory has uncommitted changes. Please commit or stash them before proceeding."
        )
        sys.exit(1)

    # Check if the current branch is 'main'
    logging.info("Checking the current branch...")
    current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    if current_branch != "main":
        logging.error(
            "You are not on the 'main' branch. Please switch to 'main' before proceeding."
        )
        sys.exit(1)

    # Pull the latest changes from remote 'main'
    logging.info("Pulling the latest changes from remote 'main'...")
    run_command("git pull origin main")


def get_latest_tag():
    try:
        latest_tag = run_command("git describe --tags --abbrev=0")
        if latest_tag.startswith("v"):
            latest_tag = latest_tag[1:]
    except subprocess.CalledProcessError:
        latest_tag = "0.0.0"  # default if no tags exist
    return latest_tag


def increment_version(current_version, increment_type):
    major, minor, patch = map(int, current_version.split("."))
    if increment_type == "patch":
        patch += 1
    elif increment_type == "minor":
        minor += 1
        patch = 0
    elif increment_type == "major":
        major += 1
        minor = 0
        patch = 0
    return f"{major}.{minor}.{patch}"


def update_version_file(version):
    version_file_path = "langfuse/version.py"
    logging.info(f"Updating version in {version_file_path} to {version}...")

    with open(version_file_path, "r") as file:
        content = file.read()

    new_content = re.sub(
        r'__version__ = "\d+\.\d+\.\d+"', f'__version__ = "{version}"', content
    )

    with open(version_file_path, "w") as file:
        file.write(new_content)

    logging.info(f"Updated version in {version_file_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Automate the release process for the Langfuse Python SDK using Poetry."
    )
    parser.add_argument(
        "increment_type",
        choices=["patch", "minor", "major"],
        help="Specify the type of version increment.",
    )
    args = parser.parse_args()

    increment_type = args.increment_type

    try:
        logging.info("Starting release process...")

        # Preliminary checks
        logging.info("Performing preliminary checks...")
        check_git_status()
        logging.info("Git status is clean, on 'main' branch, and up to date.")

        # Get the latest tag
        current_version = get_latest_tag()
        logging.info(f"Current version: v{current_version}")

        # Determine the new version
        new_version = increment_version(current_version, increment_type)
        logging.info(f"Proposed new version: v{new_version}")

        # Ask for user confirmation
        confirm = input(
            f"Do you want to proceed with the release version v{new_version}? (y/n): "
        )
        if confirm.lower() != "y":
            logging.info("Release process aborted by user.")
            sys.exit(0)

        # Step 1: Update the version
        logging.info("Step 1: Updating the version...")
        run_command(f"poetry version {new_version}")
        logging.info(f"Updated to version v{new_version}")

        # Update the version in langfuse/version.py
        update_version_file(new_version)

        # Ask for user confirmation
        confirm = input(
            f"Please check the changed files in the working tree. Proceed with releasing v{new_version}? (y/n): "
        )
        if confirm.lower() != "y":
            logging.info("Release process aborted by user.")
            sys.exit(0)

        # Step 2: Install dependencies
        logging.info("Step 2: Installing dependencies...")
        run_command("poetry install")

        # Step 3: Build the package
        logging.info("Step 3: Building the package...")
        run_command("poetry build")

        # Step 4: Commit the changes
        logging.info("Step 4: Committing the changes...")
        run_command(f'git commit -am "chore: release v{new_version}"')

        # Step 5: Push the commit
        logging.info("Step 5: Pushing the commit...")
        run_command("git push")

        # Step 6: Tag the version
        logging.info("Step 6: Tagging the version...")
        run_command(f"git tag v{new_version}")

        # Step 7: Push the tags
        logging.info("Step 7: Pushing the tags...")
        run_command("git push --tags")

        # # Step 8: Publish to PyPi
        logging.info("Step 8: Publishing to PyPi...")
        run_command("poetry publish")
        logging.info("Published to PyPi successfully.")

        # Step 9: Prompt the user to create a GitHub release
        logging.info(
            "Step 9: Please create a new release on GitHub by visiting the following URL:"
        )
        print(
            "Go to: https://github.com/langfuse/langfuse-python/releases to create the release."
        )
        logging.info("Release process completed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running command: {e.cmd}")
        logging.error(e.stderr.decode("utf-8"))
        sys.exit(1)


if __name__ == "__main__":
    main()
