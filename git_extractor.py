#!/usr/bin/env python3
"""
Git Commit Extractor

Extracts commit information and diffs for a specific author from a local git repository.
Usage: python git_extractor.py <repo_path>
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from git import Repo, Commit
from git.exc import GitCommandError, InvalidGitRepositoryError


@dataclass
class CommitInfo:
    """Data class to hold commit information"""
    hash: str
    short_hash: str
    author_name: str
    author_email: str
    date: str
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    diff: Optional[str] = None


class GitCommitExtractor:
    """Main class for extracting git commit information"""

    def __init__(self, repo_path: str, config_path: str = "config.yaml"):
        self.repo_path = Path(repo_path)
        self.config = self._load_config(config_path)
        self.repo = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            sys.exit(1)

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'author': {'name': '', 'email': ''},
            'output': {'format': 'json', 'file': 'commits_output.json', 'include_diff': True},
            'filters': {'max_commits': 100, 'date_from': None, 'date_to': None}
        }

    def _initialize_repo(self):
        """Initialize git repository"""
        try:
            if not self.repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")

            self.repo = Repo(self.repo_path)
            if self.repo.bare:
                raise InvalidGitRepositoryError("Repository is bare")

        except InvalidGitRepositoryError as e:
            print(f"Error: Invalid git repository at {self.repo_path}")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing repository: {e}")
            sys.exit(1)

    def _matches_author(self, commit: Commit) -> bool:
        """Check if commit author matches the configured author"""
        config_author = self.config['author']
        author_name = config_author.get('name', '').strip()
        author_email = config_author.get('email', '').strip()

        if not author_name and not author_email:
            print("Warning: No author name or email specified in config")
            return False

        # Check name match
        name_match = True
        if author_name:
            name_match = commit.author.name.lower() == author_name.lower()

        # Check email match
        email_match = True
        if author_email:
            email_match = commit.author.email.lower() == author_email.lower()

        return name_match and email_match

    def _is_within_date_range(self, commit: Commit) -> bool:
        """Check if commit is within the specified date range"""
        filters = self.config['filters']
        commit_date = datetime.fromtimestamp(commit.committed_date)

        date_from = filters.get('date_from')
        date_to = filters.get('date_to')

        if date_from:
            from_date = datetime.fromisoformat(date_from)
            if commit_date < from_date:
                return False

        if date_to:
            to_date = datetime.fromisoformat(date_to)
            if commit_date > to_date:
                return False

        return True

    def _get_commit_stats(self, commit: Commit) -> tuple:
        """Get insertion and deletion stats for a commit"""
        try:
            stats = commit.stats.total
            return stats.get('insertions', 0), stats.get('deletions', 0)
        except:
            return 0, 0

    def _get_commit_diff(self, commit: Commit) -> Optional[str]:
        """Get the diff for a commit"""
        if not self.config['output'].get('include_diff', True):
            return None

        try:
            # Get diff against parent commit
            if commit.parents:
                diff = self.repo.git.diff(commit.parents[0], commit, unified=3)
            else:
                # For initial commit, show diff against empty tree
                diff = self.repo.git.show(commit.hexsha, format='', unified=3)
            return diff
        except GitCommandError as e:
            print(f"Warning: Could not get diff for commit {commit.hexsha[:8]}: {e}")
            return None

    def _create_commit_info(self, commit: Commit) -> CommitInfo:
        """Create CommitInfo object from git commit"""
        insertions, deletions = self._get_commit_stats(commit)
        diff = self._get_commit_diff(commit)

        # Get list of changed files
        files_changed = []
        try:
            files_changed = list(commit.stats.files.keys())
        except:
            pass

        return CommitInfo(
            hash=commit.hexsha,
            short_hash=commit.hexsha[:8],
            author_name=commit.author.name,
            author_email=commit.author.email,
            date=datetime.fromtimestamp(commit.committed_date).isoformat(),
            message=commit.message.strip(),
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            diff=diff
        )

    def extract_commits(self) -> List[CommitInfo]:
        """Extract commits matching the criteria"""
        self._initialize_repo()

        commits_info = []
        max_commits = self.config['filters'].get('max_commits', 0)
        processed_count = 0

        print(f"Scanning repository: {self.repo_path}")
        print(f"Looking for commits by: {self.config['author']}")

        try:
            # Iterate through all commits
            for commit in self.repo.iter_commits():
                # Check if we've reached the max limit
                if max_commits > 0 and len(commits_info) >= max_commits:
                    break

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} commits, found {len(commits_info)} matches...")

                # Filter by author
                if not self._matches_author(commit):
                    continue

                # Filter by date range
                if not self._is_within_date_range(commit):
                    continue

                # Create commit info and add to list
                commit_info = self._create_commit_info(commit)
                commits_info.append(commit_info)

        except Exception as e:
            print(f"Error while processing commits: {e}")
            return commits_info

        print(f"Found {len(commits_info)} matching commits out of {processed_count} total commits")
        return commits_info

    def save_output(self, commits: List[CommitInfo]):
        """Save the extracted commits to file"""
        output_config = self.config['output']
        output_file = output_config.get('file', 'commits_output.json')
        output_format = output_config.get('format', 'json')

        if output_format.lower() == 'json':
            self._save_as_json(commits, output_file)
        else:
            self._save_as_text(commits, output_file)

    def _save_as_json(self, commits: List[CommitInfo], filename: str):
        """Save commits as JSON file"""
        data = {
            'meta': {
                'total_commits': len(commits),
                'extracted_at': datetime.now().isoformat(),
                'repository': str(self.repo_path),
                'author_filter': self.config['author']
            },
            'commits': [asdict(commit) for commit in commits]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(commits)} commits to {filename}")

    def _save_as_text(self, commits: List[CommitInfo], filename: str):
        """Save commits as text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Git Commit Extract Report\n")
            f.write(f"========================\n")
            f.write(f"Repository: {self.repo_path}\n")
            f.write(f"Author Filter: {self.config['author']}\n")
            f.write(f"Total Commits: {len(commits)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            for i, commit in enumerate(commits, 1):
                f.write(f"Commit #{i}\n")
                f.write(f"Hash: {commit.hash}\n")
                f.write(f"Author: {commit.author_name} <{commit.author_email}>\n")
                f.write(f"Date: {commit.date}\n")
                f.write(f"Files Changed: {len(commit.files_changed)}\n")
                f.write(f"Insertions: +{commit.insertions}, Deletions: -{commit.deletions}\n")
                f.write(f"Message:\n{commit.message}\n")

                if commit.files_changed:
                    f.write(f"Changed Files:\n")
                    for file_path in commit.files_changed:
                        f.write(f"  - {file_path}\n")

                if commit.diff:
                    f.write(f"\nDiff:\n")
                    f.write(commit.diff)

                f.write(f"\n{'-' * 80}\n\n")

        print(f"Saved {len(commits)} commits to {filename}")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python git_extractor.py <repo_path>")
        print("Make sure config.yaml exists in the same directory")
        sys.exit(1)

    repo_path = sys.argv[1]

    try:
        extractor = GitCommitExtractor(repo_path)
        commits = extractor.extract_commits()

        if commits:
            extractor.save_output(commits)
        else:
            print("No commits found matching the criteria")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
