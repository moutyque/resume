#!/usr/bin/env python3
"""
Resume Builder from Git Commits

Analyzes exported git commit data and generates a professional resume
highlighting the developer's contributions and technical skills.
"""
import argparse
import json
import sys

import requests
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import openai
from anthropic import Anthropic


@dataclass
class ProjectSummary:
    """Summary of work done on a project"""
    name: str
    commit_count: int
    lines_added: int
    lines_removed: int
    date_range: tuple
    primary_languages: List[str]
    key_features: List[str]
    file_types: Dict[str, int]
    commit_patterns: Dict[str, int]


class CommitAnalyzer:
    """Analyzes commit data to extract meaningful insights"""

    def __init__(self):
        # Common patterns to identify types of work
        self.commit_patterns = {
            'feature': r'(?i)\b(add|implement|create|new|feature)\b',
            'bugfix': r'(?i)\b(fix|bug|issue|resolve|patch)\b',
            'refactor': r'(?i)\b(refactor|cleanup|improve|optimize)\b',
            'test': r'(?i)\b(test|spec|unittest|testing)\b',
            'documentation': r'(?i)\b(doc|readme|comment|documentation)\b',
            'setup': r'(?i)\b(setup|config|init|install|dependency)\b',
            'performance': r'(?i)\b(performance|speed|optimize|cache)\b',
            'security': r'(?i)\b(security|auth|permission|secure)\b',
            'ui': r'(?i)\b(ui|interface|design|styling|css|frontend)\b',
            'api': r'(?i)\b(api|endpoint|rest|graphql|service)\b',
            'database': r'(?i)\b(database|db|sql|migration|schema)\b',
        }

        # Programming language detection from file extensions
        self.language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
            '.go': 'Go', '.rs': 'Rust', '.rb': 'Ruby', '.php': 'PHP',
            '.swift': 'Swift', '.kt': 'Kotlin', '.scala': 'Scala',
            '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS',
            '.jsx': 'React', '.tsx': 'React/TypeScript', '.vue': 'Vue.js',
            '.sql': 'SQL', '.yaml': 'YAML', '.yml': 'YAML',
            '.json': 'JSON', '.xml': 'XML', '.md': 'Markdown',
            '.dockerfile': 'Docker', '.tf': 'Terraform'
        }

    def analyze_project(self, project_data: Dict[str, Any]) -> ProjectSummary:
        """Analyze a single project's commit data"""
        commits = project_data.get('commits', [])
        meta = project_data.get('meta', {})

        if not commits:
            return None

        # Basic metrics
        total_additions = sum(c.get('insertions', 0) for c in commits)
        total_deletions = sum(c.get('deletions', 0) for c in commits)

        # Date range
        dates = [datetime.fromisoformat(c['date'].replace('Z', '+00:00')) for c in commits]
        date_range = (min(dates), max(dates))

        # Analyze file types and languages
        file_counter = Counter()
        language_counter = Counter()

        for commit in commits:
            for file_path in commit.get('files_changed', []):
                ext = Path(file_path).suffix.lower()
                file_counter[ext] += 1
                if ext in self.language_map:
                    language_counter[self.language_map[ext]] += 1

        # Identify commit patterns
        pattern_counter = Counter()
        key_features = []

        for commit in commits:
            message = commit.get('message', '')

            # Count pattern matches
            for pattern_name, pattern_regex in self.commit_patterns.items():
                if re.search(pattern_regex, message):
                    pattern_counter[pattern_name] += 1

            # Extract potential features (first line of significant commits)
            first_line = message.split('\n')[0].strip()
            if len(first_line) > 10 and not first_line.lower().startswith(('merge', 'wip', 'temp')):
                key_features.append(first_line)

        return ProjectSummary(
            name=meta.get('repository_name', 'Unknown Project'),
            commit_count=len(commits),
            lines_added=total_additions,
            lines_removed=total_deletions,
            date_range=date_range,
            primary_languages=language_counter.most_common(5),
            key_features=key_features[:10],  # Top 10 features
            file_types=dict(file_counter.most_common(10)),
            commit_patterns=dict(pattern_counter)
        )

    def generate_overall_summary(self, projects: List[ProjectSummary]) -> Dict[str, Any]:
        """Generate overall summary across all projects"""
        if not projects:
            return {}

        total_commits = sum(p.commit_count for p in projects)
        total_lines_added = sum(p.lines_added for p in projects)
        total_lines_removed = sum(p.lines_removed for p in projects)

        # Aggregate languages
        all_languages = Counter()
        for project in projects:
            for lang, count in project.primary_languages:
                all_languages[lang] += count

        # Aggregate patterns
        all_patterns = Counter()
        for project in projects:
            for pattern, count in project.commit_patterns.items():
                all_patterns[pattern] += count

        # Date range
        all_dates = []
        for project in projects:
            all_dates.extend(project.date_range)

        return {
            'total_projects': len(projects),
            'total_commits': total_commits,
            'total_lines_added': total_lines_added,
            'total_lines_removed': total_lines_removed,
            'date_range': (min(all_dates), max(all_dates)) if all_dates else None,
            'primary_languages': all_languages.most_common(10),
            'work_patterns': all_patterns.most_common(),
            'avg_commits_per_project': total_commits / len(projects),
            'projects_by_activity': sorted(projects, key=lambda p: p.commit_count, reverse=True)[:5]
        }



class LLMResumeGenerator:
    """Generates resume content using various LLM providers including private endpoints"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'openai').lower()
        self.session = requests.Session()

        # Set up session headers
        headers = config.get('headers', {})
        if 'api_key' in config and config['api_key']:
            # Add API key to Authorization header if not already specified
            if 'Authorization' not in headers:
                headers['Authorization'] = f"Bearer {config['api_key']}"

        self.session.headers.update(headers)
        self.timeout = config.get('timeout', 60)

        if self.provider == 'openai':
            import openai
            openai.api_key = config.get('api_key')
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            self.anthropic_client = Anthropic(api_key=config.get('api_key'))

    def generate_resume(self, summary: Dict[str, Any], projects: List[ProjectSummary]) -> Dict[str, Any]:
        """Generate resume content using LLM"""
        prompt = self.create_structured_prompt(summary, projects)

        try:
            if self.provider == 'private':
                return self._generate_with_private_endpoint(prompt)
            elif self.provider == 'openai':
                return self._generate_with_openai(prompt)
            elif self.provider == 'anthropic':
                return self._generate_with_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            print(f"Error generating resume with {self.provider}: {e}")
            return self._generate_fallback_resume(summary, projects)

    def _generate_with_private_endpoint(self, prompt: str) -> Dict[str, Any]:
        """Generate using private LLM endpoint"""
        endpoint = self.config.get('endpoint')
        if not endpoint:
            raise ValueError("Private endpoint URL not configured")

        # Format payload based on common API patterns
        # This supports both OpenAI-compatible and custom formats
        payload = {
            "model": self.config.get('model', 'default'),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional resume writer with expertise in technical resumes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.get('max_tokens', 2000),
            "temperature": self.config.get('temperature', 0.7),
            "stream": False
        }

        print(f"🔗 Sending request to private endpoint: {endpoint}")

        try:
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            content = self._extract_content_from_response(result)

            # Parse JSON response
            return self._parse_llm_response(content)

        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse response JSON: {e}")

    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from various response formats"""
        # Try OpenAI-compatible format first
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            if 'message' in choice:
                return choice['message'].get('content', '')
            elif 'text' in choice:
                return choice['text']

        # Try direct content fields
        if 'content' in response:
            return response['content']

        if 'response' in response:
            return response['response']

        if 'text' in response:
            return response['text']

        # Try to find any string field that looks like generated content
        for key, value in response.items():
            if isinstance(value, str) and len(value) > 100:
                return value

        raise ValueError(f"Could not extract content from response: {response}")

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate LLM response content"""
        content = content.strip()

        # Try to extract JSON from markdown code blocks
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            if json_end > json_start:
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content[json_start:].strip()
        elif '```' in content:
            # Handle generic code blocks
            json_start = content.find('```') + 3
            json_end = content.rfind('```')
            if json_end > json_start:
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content[json_start:].strip()
        else:
            # Try to find JSON-like content
            if content.startswith('{') and content.endswith('}'):
                json_str = content
            else:
                # Extract JSON from mixed content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                else:
                    raise ValueError("No valid JSON found in response")

        try:
            parsed = json.loads(json_str)

            # Validate required fields
            required_fields = ['professional_summary', 'technical_skills', 'projects', 'key_accomplishments']
            for field in required_fields:
                if field not in parsed:
                    print(f"⚠️  Warning: Missing required field '{field}' in LLM response")
                    parsed[field] = []

            return parsed

        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse JSON response, using fallback. Error: {e}")
            print(f"Raw response: {content[:500]}...")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def _generate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate using OpenAI"""
        import openai

        response = openai.ChatCompletion.create(
            model=self.config.get('model', 'gpt-4'),
            messages=[
                {"role": "system", "content": "You are a professional resume writer with expertise in technical resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 2000)
        )

        content = response.choices[0].message.content.strip()
        return self._parse_llm_response(content)

    def _generate_with_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Generate using Anthropic Claude"""
        response = self.anthropic_client.completion(
            model=self.config.get('model', 'claude-3-sonnet'),
            prompt=f"\n\nHuman: {prompt}\n\nAssistant: I'll analyze this git commit data and create a professional resume. Here's the structured response:\n\n```json",
            max_tokens_to_sample=self.config.get('max_tokens', 2000),
            temperature=self.config.get('temperature', 0.7)
        )

        content = response.completion.strip()
        if content.endswith('```'):
            content = content[:-3]

        return json.loads(content)

    def _generate_fallback_resume(self, summary: Dict[str, Any], projects: List[ProjectSummary]) -> Dict[str, Any]:
        """Generate basic resume if LLM fails"""
        languages = [lang for lang, _ in summary.get('primary_languages', [])]

        return {
            "professional_summary": f"Experienced software developer with contributions to {summary.get('total_projects', 0)} projects and {summary.get('total_commits', 0)} commits. Proficient in {', '.join(languages[:3])} with focus on feature development and code quality.",
            "technical_skills": languages[:10],
            "projects": [
                {
                    "name": project.name,
                    "description": [
                        f"Contributed {project.commit_count} commits with {project.lines_added} lines of code",
                        f"Worked with {', '.join([lang for lang, _ in project.primary_languages[:2]])}"
                    ]
                } for project in projects[:5]
            ],
            "key_accomplishments": [
                f"Maintained active development across {len(projects)} projects",
                f"Contributed over {summary.get('total_lines_added', 0)} lines of production code",
            ]
        }

    def create_structured_prompt(self, summary: Dict[str, Any], projects: List[ProjectSummary]) -> str:
        """Create a structured prompt for the LLM"""

        # Format project details
        project_details = []
        for project in projects:
            duration_days = (project.date_range[1] - project.date_range[0]).days
            project_details.append(f"""
Project: {project.name}
- Commits: {project.commit_count}
- Duration: {duration_days} days ({project.date_range[0].strftime('%Y-%m')} to {project.date_range[1].strftime('%Y-%m')})
- Code changes: +{project.lines_added} -{project.lines_removed} lines
- Primary technologies: {', '.join([lang for lang, _ in project.primary_languages[:3]])}
- Work focus: {', '.join([f'{k} ({v})' for k, v in list(project.commit_patterns.items())[:3]])}
- Key implementations: {'; '.join(project.key_features[:5])}
""")

        prompt = f"""
You are a professional resume writer. Based on the following git commit analysis, create a compelling professional summary and project descriptions for a software developer's resume.

OVERALL DEVELOPER PROFILE:
- Total projects worked on: {summary.get('total_projects', 0)}
- Total commits: {summary.get('total_commits', 0)}
- Code contribution: +{summary.get('total_lines_added', 0)} -{summary.get('total_lines_removed', 0)} lines
- Active period: {summary.get('date_range', ('Unknown', 'Unknown'))[0].strftime('%Y-%m') if summary.get('date_range') else 'Unknown'} to {summary.get('date_range', ('Unknown', 'Unknown'))[1].strftime('%Y-%m') if summary.get('date_range') else 'Unknown'}
- Primary technologies: {', '.join([lang for lang, _ in summary.get('primary_languages', [])[:5]])}
- Work distribution: {', '.join([f'{k}: {v}%' for k, v in [(k, round(v/summary.get('total_commits', 1)*100)) for k, v in summary.get('work_patterns', [])][:5]])}

PROJECT DETAILS:
{''.join(project_details)}

REQUIREMENTS:
1. Create a professional summary (3-4 sentences) highlighting the developer's expertise and impact
2. For each project, write a 2-3 bullet point description focusing on:
   - Technical achievements and implementations
   - Technologies used and problems solved
   - Quantifiable impact where possible
3. Identify and list the top 8-10 technical skills based on commit patterns
4. Suggest 3-5 key accomplishments that would be impressive to employers
5. Use action verbs and quantify achievements where possible
6. Keep the tone professional but engaging

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
    "professional_summary": "...",
    "technical_skills": ["skill1", "skill2", ...],
    "projects": [
        {{
            "name": "Project Name",
            "description": ["bullet point 1", "bullet point 2", ...]
        }}
    ],
    "key_accomplishments": ["accomplishment 1", "accomplishment 2", ...],
    "recommendations": ["suggestion 1", "suggestion 2", ...] // Optional suggestions for resume improvement
}}
"""
        return prompt

class ResumeBuilder:
    """Main class for building resumes from git data"""

    def __init__(self, config_path: str = "resume_config.yaml"):
        """
        Initialize the resume builder

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.analyzer = CommitAnalyzer()
        self.llm_generator = LLMResumeGenerator(self.config['llm'])

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ['input', 'output', 'llm', 'analysis']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")

            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'input': {
                'data_directory': 'output',
                'file_pattern': '*_commits.json'
            },
            'output': {
                'filename': 'generated_resume.json',
                'include_analysis': True,
                'format': 'json'  # Could extend to markdown, pdf later
            },
            'llm': {
                'provider': 'openai',  # or 'anthropic'
                'model': 'gpt-4',
                'api_key': 'your-api-key-here'
            },
            'analysis': {
                'min_commits_per_project': 5,
                'exclude_projects': [],
                'focus_recent_months': 12
            }
        }

    def load_commit_data(self) -> List[Dict[str, Any]]:
        """Load all commit data files"""
        input_config = self.config['input']
        data_dir = Path(input_config['data_directory'])
        pattern = input_config['file_pattern']

        commit_data = []
        for file_path in data_dir.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    commit_data.append(data)
                print(f"Loaded {len(data.get('commits', []))} commits from {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return commit_data

    def build_resume(self):
        """Main method to build the resume"""
        print("🔍 Loading commit data...")
        commit_data = self.load_commit_data()

        if not commit_data:
            print("❌ No commit data found. Please run git_extractor.py first.")
            return

        print(f"📊 Analyzing {len(commit_data)} projects...")

        # Analyze each project
        projects = []
        for data in commit_data:
            project_summary = self.analyzer.analyze_project(data)
            if project_summary and project_summary.commit_count >= self.config['analysis']['min_commits_per_project']:
                projects.append(project_summary)

        if not projects:
            print("❌ No projects meet the minimum commit threshold.")
            return

        # Generate overall summary
        overall_summary = self.analyzer.generate_overall_summary(projects)

        print(f"🤖 Generating resume with {self.config['llm']['provider']}...")

        # Generate resume using LLM
        resume_content = self.llm_generator.generate_resume(overall_summary, projects)

        # Prepare final output
        final_output = {
            'resume': resume_content,
            'generated_at': datetime.now().isoformat(),
            'source_analysis': {
                'projects_analyzed': len(projects),
                'total_commits': overall_summary.get('total_commits', 0),
                'date_range': [
                    overall_summary['date_range'][0].isoformat() if overall_summary.get('date_range') else None,
                    overall_summary['date_range'][1].isoformat() if overall_summary.get('date_range') else None
                ],
                'primary_languages': overall_summary.get('primary_languages', [])
            }
        }

        if self.config['output']['include_analysis']:
            final_output['detailed_analysis'] = {
                'overall_summary': overall_summary,
                'project_summaries': [
                    {
                        'name': p.name,
                        'commits': p.commit_count,
                        'lines_changed': {'added': p.lines_added, 'removed': p.lines_removed},
                        'languages': p.primary_languages,
                        'patterns': p.commit_patterns
                    } for p in projects
                ]
            }

        # Save output
        output_file = Path(self.config['output']['filename'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Resume generated and saved to {output_file}")
        print(f"📈 Analyzed {len(projects)} projects with {overall_summary.get('total_commits', 0)} commits")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate a professional resume from git commit data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default config (resume_config.yaml)
  %(prog)s -c custom_config.yaml              # Use custom config file
  %(prog)s --config /path/to/config.yaml      # Use config with full path
  %(prog)s -c config.yaml -v                  # Verbose output
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='resume_config.yaml',
        help='Path to configuration file (default: resume_config.yaml)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze data without calling LLM (for testing)'
    )

    return parser.parse_args()

def validate_config(config_path: str) -> bool:
    """Validate configuration file"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['input', 'output', 'llm', 'analysis']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False

        # Check input directory exists
        input_dir = Path(config['input']['data_directory'])
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False

        # Check LLM configuration
        llm_config = config['llm']
        if llm_config.get('provider') == 'private':
            if not llm_config.get('endpoint'):
                print("❌ Private LLM provider requires 'endpoint' configuration")
                return False

        print(f"✅ Configuration file is valid: {config_path}")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def main():
    """Main entry point for the resume builder"""
    args = parse_arguments()

    # Configure verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        # Validate configuration if requested
        if args.validate_config:
            is_valid = validate_config(args.config)
            sys.exit(0 if is_valid else 1)

        # Build resume
        builder = ResumeBuilder(config_path=args.config)
        resume_data = builder.build_resume()

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
