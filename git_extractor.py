#!/usr/bin/env python3
"""
Git Commit Extractor

Extracts commit information and diffs for a specific author from multiple local git repositories.
Usage: python git_extractor.py [repo_path] (if repo_path provided, ignores config repositories)
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


@dataclass
class RepositoryConfig:
    """Data class for repository configuration"""
    path: str
    name: Optional[str] = None


class GitCommitExtractor:
    """Main class for extracting git commit information"""

    def __init__(self, config_path: str = "config.yaml", single_repo: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.repositories = self._parse_repositories(single_repo)
        self._ensure_output_directory()

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
            'repositories': [],
            'author': {'name': '', 'email': ''},
            'output': {
                'format': 'json',
                'directory': 'output',
                'filename_template': '{name}_commits.{ext}',
                'include_diff': True
            },
            'filters': {'max_commits': 100, 'date_from': None, 'date_to': None}
        }

    def _parse_repositories(self, single_repo: Optional[str]) -> List[RepositoryConfig]:
        """Parse repository configurations from config or single repo argument"""
        repositories = []

        # If single repo provided via argument, use that instead of config
        if single_repo:
            repo_path = Path(single_repo)
            repo_name = repo_path.name
            return [RepositoryConfig(path=single_repo, name=repo_name)]

        # Parse repositories from config
        repo_configs = self.config.get('repositories', [])
        if not repo_configs:
            print("No repositories specified in config file")
            return []

        for repo_config in repo_configs:
            if isinstance(repo_config, str):
                # Simple string path
                repo_path = Path(repo_config)
                repositories.append(RepositoryConfig(
                    path=repo_config,
                    name=repo_path.name
                ))
            elif isinstance(repo_config, dict):
                # Dictionary with path and optional name
                path = repo_config.get('path')
                if not path:
                    print(f"Warning: Repository config missing 'path': {repo_config}")
                    continue

                name = repo_config.get('name')
                if not name:
                    name = Path(path).name

                repositories.append(RepositoryConfig(path=path, name=name))
            else:
                print(f"Warning: Invalid repository config: {repo_config}")

        return repositories

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_dir = Path(self.config['output'].get('directory', 'output'))
        output_dir.mkdir(exist_ok=True)

    def _initialize_repo(self, repo_path: str) -> Optional[Repo]:
        """Initialize git repository"""
        try:
            path = Path(repo_path)
            if not path.exists():
                print(f"Warning: Repository path does not exist: {repo_path}")
                return None

            repo = Repo(repo_path)
            if repo.bare:
                print(f"Warning: Repository is bare: {repo_path}")
                return None

            return repo

        except InvalidGitRepositoryError as e:
            print(f"Warning: Invalid git repository at {repo_path}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error initializing repository {repo_path}: {e}")
            return None

    def _matches_author(self, commit: Commit) -> bool:
        """Check if commit author matches the configured author"""
        config_author = self.config['author']
        author_name = config_author.get('name', '').strip()
        author_email = config_author.get('email', '').strip()

        if not author_name and not author_email:
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

    def _get_commit_diff(self, commit: Commit, repo: Repo) -> Optional[str]:
        """Get the diff for a commit"""
        if not self.config['output'].get('include_diff', True):
            return None

        try:
            # Get diff against parent commit
            if commit.parents:
                diff = repo.git.diff(commit.parents[0], commit, unified=3)
            else:
                # For initial commit, show diff against empty tree
                diff = repo.git.show(commit.hexsha, format='', unified=3)
            return diff
        except GitCommandError as e:
            print(f"Warning: Could not get diff for commit {commit.hexsha[:8]}: {e}")
            return None

    def _create_commit_info(self, commit: Commit, repo: Repo) -> CommitInfo:
        """Create CommitInfo object from git commit"""
        insertions, deletions = self._get_commit_stats(commit)
        diff = self._get_commit_diff(commit, repo)

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

    def extract_commits_from_repo(self, repo_config: RepositoryConfig) -> List[CommitInfo]:
        """Extract commits from a single repository"""
        print(f"\n{'='*60}")
        print(f"Processing repository: {repo_config.name}")
        print(f"Path: {repo_config.path}")
        print(f"{'='*60}")

        repo = self._initialize_repo(repo_config.path)
        if not repo:
            print(f"Skipping repository: {repo_config.name}")
            return []

        commits_info = []
        max_commits = self.config['filters'].get('max_commits', 0)
        processed_count = 0

        print(f"Looking for commits by: {self.config['author']}")

        try:
            # Iterate through all commits
            for commit in repo.iter_commits():
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
                commit_info = self._create_commit_info(commit, repo)
                commits_info.append(commit_info)

        except Exception as e:
            print(f"Error while processing commits in {repo_config.name}: {e}")

        print(f"Found {len(commits_info)} matching commits out of {processed_count} total commits")
        return commits_info

    def _generate_filename(self, repo_name: str) -> str:
        """Generate output filename based on template"""
        output_config = self.config['output']
        template = output_config.get('filename_template', '{name}_commits.{ext}')
        ext = 'json' if output_config.get('format', 'json').lower() == 'json' else 'txt'

        filename = template.format(
            name=repo_name,
            date=datetime.now().strftime('%Y%m%d'),
            ext=ext
        )

        output_dir = Path(output_config.get('directory', 'output'))
        return str(output_dir / filename)

    def save_output(self, commits: List[CommitInfo], repo_config: RepositoryConfig):
        """Save the extracted commits to file"""
        if not commits:
            print(f"No commits to save for {repo_config.name}")
            return

        output_format = self.config['output'].get('format', 'json')
        filename = self._generate_filename(repo_config.name)

        if output_format.lower() == 'json':
            self._save_as_json(commits, filename, repo_config)
        else:
            print(f"Can not save unknown export format {output_format.lower}")
    def _save_as_json(self, commits: List[CommitInfo], filename: str, repo_config: RepositoryConfig):
        """Save commits as JSON file"""
        data = {
            'meta': {
                'repository_name': repo_config.name,
                'repository_path': repo_config.path,
                'total_commits': len(commits),
                'extracted_at': datetime.now().isoformat(),
                'author_filter': self.config['author']
            },
            'commits': [asdict(commit) for commit in commits]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(commits)} commits to {filename}")


    def process_all_repositories(self):
        """Process all configured repositories"""
        if not self.repositories:
            print("No repositories to process")
            return

        print(f"Starting extraction from {len(self.repositories)} repositories...")
        print(f"Author filter: {self.config['author']}")
        print(f"Output directory: {self.config['output'].get('directory', 'output')}")

        total_commits = 0
        successful_repos = 0

        for repo_config in self.repositories:
            try:
                commits = self.extract_commits_from_repo(repo_config)
                if commits:
                    self.save_output(commits, repo_config)
                    total_commits += len(commits)
                    successful_repos += 1
                else:
                    print(f"No commits found for {repo_config.name}")

            except KeyboardInterrupt:
                print(f"\nOperation cancelled by user")
                break
            except Exception as e:
                print(f"Error processing repository {repo_config.name}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Repositories processed: {successful_repos}/{len(self.repositories)}")
        print(f"Total commits extracted: {total_commits}")
        print(f"Output directory: {self.config['output'].get('directory', 'output')}")


def main():
    """Main function"""
    single_repo = None

    # Check if a single repository path was provided as argument
    if len(sys.argv) == 2:
        single_repo = sys.argv[1]
    elif len(sys.argv) > 2:
        print("Usage: python git_extractor.py [repo_path]")
        print("If repo_path is provided, it will process only that repository.")
        print("Otherwise, it will process all repositories listed in config.yaml")
        sys.exit(1)

    try:
        extractor = GitCommitExtractor(single_repo=single_repo)
        extractor.process_all_repositories()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
