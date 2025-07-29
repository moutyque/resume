#!/usr/bin/env python3
"""
Git Commit Extractor

Automatically discovers and extracts commit information from all git repositories
under a base directory for a specific author.
Usage: python git_extractor.py [repo_path] (if repo_path provided, processes single repo)
"""

import sys
import json
import yaml
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
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
class RepositoryInfo:
    """Data class for repository information"""
    path: Path
    name: str
    relative_path: str


class GitRepositoryScanner:
    """Class for scanning and discovering git repositories"""

    def __init__(self, base_path: str, config: Dict[str, Any]):
        self.base_path = Path(base_path).resolve()
        self.config = config
        self.scanning_config = config.get('scanning', {})

    def _is_excluded(self, path: Path) -> bool:
        """Check if path matches any exclusion patterns"""
        exclude_patterns = self.scanning_config.get('exclude_patterns', [])

        for pattern in exclude_patterns:
            # Check if any part of the path matches the pattern
            for part in path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

            # Also check the full relative path
            try:
                relative_path = path.relative_to(self.base_path)
                if fnmatch.fnmatch(str(relative_path), pattern):
                    return True
            except ValueError:
                pass

        return False

    def _is_git_repository(self, path: Path) -> bool:
        """Check if a directory is a git repository"""
        try:
            repo = Repo(str(path))

            # Check if we should include bare repositories
            if repo.bare and not self.scanning_config.get('include_bare', False):
                return False

            return True
        except (InvalidGitRepositoryError, Exception):
            return False

    def _scan_directory(self, current_path: Path, current_depth: int, max_depth: int) -> List[RepositoryInfo]:
        """Recursively scan directory for git repositories"""
        repositories = []

        # Check depth limit
        if 0 < max_depth < current_depth:
            return repositories

        try:
            # Check if current directory is excluded
            if self._is_excluded(current_path):
                return repositories

            # Check if current directory is a git repository
            if self._is_git_repository(current_path):
                relative_path = current_path.relative_to(self.base_path)
                repo_name = current_path.name

                repositories.append(RepositoryInfo(
                    path=current_path,
                    name=repo_name,
                    relative_path=str(relative_path)
                ))

                # If it's a git repo, don't scan its subdirectories
                # (avoid scanning .git subdirectories and nested repos)
                return repositories

            # Scan subdirectories
            if current_path.is_dir():
                try:
                    for item in current_path.iterdir():
                        if item.is_dir() and not item.name.startswith('.'):
                            sub_repos = self._scan_directory(item, current_depth + 1, max_depth)
                            repositories.extend(sub_repos)
                except PermissionError:
                    print(f"Permission denied accessing: {current_path}")
                except Exception as e:
                    print(f"Error scanning directory {current_path}: {e}")

        except Exception as e:
            print(f"Error processing {current_path}: {e}")

        return repositories

    def discover_repositories(self) -> List[RepositoryInfo]:
        """Discover all git repositories under the base path"""
        if not self.base_path.exists():
            print(f"Base path does not exist: {self.base_path}")
            return []

        if not self.base_path.is_dir():
            print(f"Base path is not a directory: {self.base_path}")
            return []

        print(f"Scanning for git repositories under: {self.base_path}")
        max_depth = self.scanning_config.get('max_depth', 2)
        print(f"Maximum scan depth: {max_depth if max_depth > 0 else 'unlimited'}")

        if self.scanning_config.get('exclude_patterns'):
            print(f"Excluding patterns: {', '.join(self.scanning_config['exclude_patterns'])}")

        repositories = self._scan_directory(self.base_path, 1, max_depth)

        print(f"Found {len(repositories)} git repositories")
        return sorted(repositories, key=lambda r: r.relative_path)


class GitCommitExtractor:
    """Main class for extracting git commit information"""

    def __init__(self, config_path: str = "config.yaml", single_repo: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.repositories = self._discover_repositories(single_repo)
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
            'scanning': {
                'base_path': '.',
                'max_depth': 2,
                'exclude_patterns': ['node_modules', '.venv', 'venv', '__pycache__'],
                'include_bare': False
            },
            'author': {'name': '', 'email': ''},
            'output': {
                'format': 'json',
                'directory': 'output',
                'filename_template': '{name}_commits.{ext}',
                'include_diff': True
            },
            'filters': {'max_commits': 100, 'date_from': None, 'date_to': None}
        }

    def _discover_repositories(self, single_repo: Optional[str]) -> List[RepositoryInfo]:
        """Discover repositories either from single path or by scanning base path"""
        if single_repo:
            # Process single repository provided via argument
            repo_path = Path(single_repo).resolve()
            return [RepositoryInfo(
                path=repo_path,
                name=repo_path.name,
                relative_path=repo_path.name
            )]

        # Discover repositories by scanning base path
        base_path = self.config['scanning'].get('base_path', '.')
        if not base_path:
            print("No base_path specified in scanning configuration")
            return []

        scanner = GitRepositoryScanner(base_path, self.config)
        return scanner.discover_repositories()

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_dir = Path(self.config['output'].get('directory', 'output'))
        output_dir.mkdir(exist_ok=True)

    def _initialize_repo(self, repo_info: RepositoryInfo) -> Optional[Repo]:
        """Initialize git repository"""
        try:
            repo = Repo(str(repo_info.path))
            return repo

        except InvalidGitRepositoryError as e:
            print(f"Warning: Invalid git repository at {repo_info.path}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error initializing repository {repo_info.path}: {e}")
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

    def extract_commits_from_repo(self, repo_info: RepositoryInfo) -> List[CommitInfo]:
        """Extract commits from a single repository"""
        print(f"\n{'='*80}")
        print(f"Processing repository: {repo_info.name}")
        print(f"Path: {repo_info.path}")
        print(f"Relative path: {repo_info.relative_path}")
        print(f"{'='*80}")

        repo = self._initialize_repo(repo_info)
        if not repo:
            print(f"Skipping repository: {repo_info.name}")
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
                    print(f"  Processed {processed_count} commits, found {len(commits_info)} matches...")

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
            print(f"Error while processing commits in {repo_info.name}: {e}")

        print(f"✓ Found {len(commits_info)} matching commits out of {processed_count} total commits")
        return commits_info

    def _generate_filename(self, repo_info: RepositoryInfo) -> str:
        """Generate output filename based on template"""
        output_config = self.config['output']
        template = output_config.get('filename_template', '{name}_commits.{ext}')
        ext = 'json' if output_config.get('format', 'json').lower() == 'json' else 'txt'

        # Use relative path as name if it provides more context
        name = repo_info.name
        if len(repo_info.relative_path.split('/')) > 1:
            # Replace path separators with underscores for filename
            name = repo_info.relative_path.replace('/', '_').replace('\\', '_')

        filename = template.format(
            name=name,
            date=datetime.now().strftime('%Y%m%d'),
            ext=ext
        )

        output_dir = Path(output_config.get('directory', 'output'))
        return str(output_dir / filename)

    def save_output(self, commits: List[CommitInfo], repo_info: RepositoryInfo):
        """Save the extracted commits to file"""
        if not commits:
            print(f"  No commits to save for {repo_info.name}")
            return

        output_format = self.config['output'].get('format', 'json')
        filename = self._generate_filename(repo_info)

        if output_format.lower() == 'json':
            self._save_as_json(commits, filename, repo_info)
        else:
            self._save_as_text(commits, filename, repo_info)

    def _save_as_json(self, commits: List[CommitInfo], filename: str, repo_info: RepositoryInfo):
        """Save commits as JSON file"""
        data = {
            'meta': {
                'repository_name': repo_info.name,
                'repository_path': str(repo_info.path),
                'relative_path': repo_info.relative_path,
                'total_commits': len(commits),
                'extracted_at': datetime.now().isoformat(),
                'author_filter': self.config['author']
            },
            'commits': [asdict(commit) for commit in commits]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved {len(commits)} commits to {filename}")

    def _save_as_text(self, commits: List[CommitInfo], filename: str, repo_info: RepositoryInfo):
        """Save commits as text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Git Commit Extract Report\n")
            f.write(f"========================\n")
            f.write(f"Repository Name: {repo_info.name}\n")
            f.write(f"Repository Path: {repo_info.path}\n")
            f.write(f"Relative Path: {repo_info.relative_path}\n")
            f.write(f"Author Filter: {self.config['author']}\n")
            f.write(f"Total Commits: {len(commits)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            for i, commit in enumerate(commits, 1):
                f.write(f"Commit #{i}\n")
                f.write(f"{'='*60}\n")
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

        print(f"  ✓ Saved {len(commits)} commits to {filename}")

    def process_all_repositories(self):
        """Process all discovered repositories"""
        if not self.repositories:
            print("No repositories to process")
            return

        print(f"\n{'='*80}")
        print(f"STARTING EXTRACTION")
        print(f"{'='*80}")
        print(f"Repositories found: {len(self.repositories)}")
        print(f"Author filter: {self.config['author']}")
        print(f"Output directory: {self.config['output'].get('directory', 'output')}")

        # Show discovered repositories
        print(f"\nDiscovered repositories:")
        for i, repo in enumerate(self.repositories, 1):
            print(f"  {i:2d}. {repo.relative_path} ({repo.name})")

        total_commits = 0
        successful_repos = 0

        for repo_info in self.repositories:
            try:
                commits = self.extract_commits_from_repo(repo_info)
                if commits:
                    self.save_output(commits, repo_info)
                    total_commits += len(commits)
                    successful_repos += 1
                else:
                    print(f"  No matching commits found for {repo_info.name}")

            except KeyboardInterrupt:
                print(f"\nOperation cancelled by user")
                break
            except Exception as e:
                print(f"  Error processing repository {repo_info.name}: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Repositories processed: {successful_repos}/{len(self.repositories)}")
        print(f"Total commits extracted: {total_commits}")
        print(f"Output directory: {self.config['output'].get('directory', 'output')}")
        if total_commits > 0:
            print(f"Average commits per repo: {total_commits/successful_repos:.1f}")


def main():
    """Main function"""
    single_repo = None

    # Check if a single repository path was provided as argument
    if len(sys.argv) == 2:
        single_repo = sys.argv[1]
    elif len(sys.argv) > 2:
        print("Usage: python git_extractor.py [repo_path]")
        print("If repo_path is provided, it will process only that repository.")
        print("Otherwise, it will scan base_path from config.yaml for git repositories.")
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
