import os
import git
import pandas as pd
import torch
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv
from github import Github
import json
import pickle

class CommitMessageResearch:
    def __init__(self, repo_url, use_local_repo=False):
        self.repo_url = repo_url
        self.repo = None
        self.model = None
        self.tokenizer = None
        self.github = None
        self.github_repo = None
        self.use_local_repo = use_local_repo
        
    def setup(self):
        """Initialize the research environment"""
        # Load environment variables
        load_dotenv()
        
        # Initialize GitHub API
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            raise ValueError("GitHub token not found in environment variables")
        
        print("Initializing GitHub API...")
        self.github = Github(github_token)
        self.github_repo = self.github.get_repo('tensorflow/tensorflow')
        
        # Initialize model
        print("Initializing model...")
        self.model_name = "codellama/CodeLlama-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test the model with a simple prompt
        print("\nTesting model with simple prompt...")
        test_prompt = "Write a commit message for adding a new feature:"
        print(f"Test prompt: {test_prompt}")
        
        print("Tokenizing test prompt...")
        test_inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.model.device)
        print(f"Input tensor shape: {test_inputs['input_ids'].shape}")
        
        print("Generating test response...")
        test_outputs = self.model.generate(
            **test_inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        print(f"Output tensor shape: {test_outputs.shape}")
        
        test_response = self.tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        print(f"Test response: {test_response}")
        print("Model test complete!\n")
        
        # Clone or update repository only if needed
        if self.use_local_repo:
            repo_name = self.repo_url.split('/')[-1].replace('.git', '')
            if os.path.exists(repo_name):
                print(f"Repository {repo_name} already exists, updating...")
                self.repo = git.Repo(repo_name)
                self.repo.remotes.origin.pull()
            else:
                print(f"Cloning repository {repo_name}...")
                self.repo = git.Repo.clone_from(self.repo_url, repo_name)
        else:
            print("Skipping repository clone (using API-only mode)")
        
        print("Setup complete!")
    
    def extract_commits(self, start_date=None, end_date=None):
        """Extract commits within a date range using GitHub API"""
        commits = []
        total_commits = 0
        filtered_by_date = 0
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            print(f"Using start date: {start_date}")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            print(f"Using end date: {end_date}")
        
        # Get commits using GitHub API with pagination and date filtering
        print("Fetching commits from GitHub API...")
        api_commits = list(self.github_repo.get_commits(
            since=start_date,
            until=end_date
        ).get_page(0))[:2]  # Limit to 2 commits
        print(f"Fetched {len(api_commits)} commits from API")
        
        for commit in tqdm(api_commits, desc="Processing commits"):
            total_commits += 1
            commit_date = commit.commit.author.date
            
            try:
                if self.use_local_repo:
                    # Get detailed diff from local repo
                    git_commit = self.repo.commit(commit.sha)
                    if git_commit.parents:
                        diff = git_commit.diff(git_commit.parents[0])
                    else:
                        diff = git_commit.diff(None)
                else:
                    # Get diff from GitHub API
                    diff = commit.files
                
                # Construct commit URL
                commit_url = f"https://github.com/{self.github_repo.full_name}/commit/{commit.sha}"
                
                commits.append({
                    'hash': commit.sha,
                    'message': commit.commit.message,
                    'date': commit_date,
                    'diff': diff,
                    'author': commit.commit.author.name,
                    'files_changed': len(commit.files),
                    'url': commit_url
                })
                print(f"Added commit: {commit.sha[:8]} - {commit.commit.message[:50]}...")
                    
            except Exception as e:
                print(f"Error processing commit {commit.sha}: {str(e)}")
                continue
        
        print(f"\nSummary:")
        print(f"Total commits fetched: {total_commits}")
        print(f"Commits processed: {len(commits)}")
        
        return pd.DataFrame(commits)
    
    def generate_commit_message(self, diff):
        """Generate commit message using CodeLlama"""
        # Convert diff object to string representation
        if self.use_local_repo:
            diff_text = "\n".join([d.diff.decode('utf-8', errors='ignore') for d in diff])
        else:
            # Format API diff information more concisely
            diff_text = "\n".join([
                f"File: {f.filename}\n"
                f"Changes: +{f.additions} -{f.deletions}\n"
                f"{f.patch[:200] if f.patch else 'No patch available'}"  # More aggressive truncation
                for f in diff[:5]  # Only take first 5 files
            ])
        
        # Truncate diff text to fit model's context window
        max_chars = 1000  # More aggressive truncation
        if len(diff_text) > max_chars:
            # Keep only the start of the diff
            diff_text = diff_text[:max_chars] + "\n...[truncated]..."
        
        prompt = f"""<s>[INST] <<SYS>>
        You are a professional software developer writing a commit message. Analyze the following code changes and write a clear, concise commit message that:
        1. Starts with a short summary (50-72 characters)
        2. Describes what was changed and why
        3. Uses present tense and imperative mood
        4. Follows conventional commit format if applicable
        <</SYS>>

        Code changes:
        {diff_text}

        Commit message: [/INST]"""
        
        print("\n" + "="*80)
        print("PROMPT:")
        print("-"*80)
        print(prompt)
        print("="*80)
        
        try:
            print("Tokenizing prompt...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            print("Generating response...")
            # Generate in smaller chunks to show progress
            current_output = ""
            for i in range(0, 100, 20):  # Generate in chunks of 20 tokens
                print(f"Generating tokens {i} to {i+20}...")
                chunk_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    num_beams=4,
                    early_stopping=True
                )
                chunk_text = self.tokenizer.decode(chunk_outputs[0], skip_special_tokens=True)
                current_output += chunk_text[len(current_output):]
                print(f"Current output: {current_output}")
            
            print("\nFinal response:")
            print("-"*80)
            print(current_output)
            print("="*80 + "\n")
            
            # Extract just the generated message (after the prompt)
            message = current_output[len(prompt):].strip()
            
            # Clean up the generated message
            # Try to extract message between triple backticks if present
            if '```' in message:
                parts = message.split('```')
                if len(parts) >= 3:  # We have content between backticks
                    message = parts[1].strip()
            
            # Remove common prefixes
            prefixes_to_remove = [
                "Here is a potential commit message for the given code changes:",
                "Here is a possible commit message for the given changes:",
                "Here is a possible commit message for the changes you described:",
                "Summary: ",
                "Commit message: "
            ]
            for prefix in prefixes_to_remove:
                if message.startswith(prefix):
                    message = message[len(prefix):].strip()
            
            # Take first non-empty line that's not a separator and not a bullet point
            for line in message.split('\n'):
                line = line.strip()
                if line and not all(c in '-=' for c in line) and not line.startswith('*'):
                    message = line
                    break
            
            return message
            
        except Exception as e:
            print(f"Error generating message: {str(e)}")
            return f"[Generation Error: {str(e)[:100]}...]"
    
    def calculate_similarity(self, msg1, msg2):
        """Calculate similarity between two messages using improved text analysis"""
        # Skip error messages
        if msg1.startswith("[Generation Error") or msg2.startswith("[Generation Error"):
            return 0.0
            
        # Normalize messages
        msg1 = msg1.lower().strip()
        msg2 = msg2.lower().strip()
        
        # Remove common prefixes and suffixes
        prefixes = ["[", "(", "fix:", "feat:", "chore:", "docs:", "test:", "style:", "refactor:"]
        for prefix in prefixes:
            if msg1.startswith(prefix):
                msg1 = msg1[msg1.find("]")+1:].strip() if "]" in msg1 else msg1[len(prefix):].strip()
            if msg2.startswith(prefix):
                msg2 = msg2[msg2.find("]")+1:].strip() if "]" in msg2 else msg2[len(prefix):].strip()
        
        # Get word sets
        words1 = set(msg1.split())
        words2 = set(msg2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_messages(self, df):
        """Compare and analyze commit messages with their diffs"""
        results = []
        for _, row in tqdm(df.iterrows(), desc="Analyzing messages"):
            # Get the diff summary
            if self.use_local_repo:
                diff_summary = "\n".join([d.diff.decode('utf-8', errors='ignore') for d in row['diff']])
            else:
                diff_summary = "\n".join([
                    f"File: {f.filename}\n"
                    f"Changes: +{f.additions} -{f.deletions}\n"
                    f"{f.patch[:200] if f.patch else 'No patch available'}"
                    for f in row['diff'][:5]
                ])

            # Generate new message based on diff
            generated_msg = self.generate_commit_message(row['diff'])
            
            # Calculate similarity
            similarity = self.calculate_similarity(row['message'], generated_msg)
            
            # Extract key information from diff
            files_modified = row['files_changed']
            if not self.use_local_repo:
                total_additions = sum(f.additions for f in row['diff'])
                total_deletions = sum(f.deletions for f in row['diff'])
                file_types = [os.path.splitext(f.filename)[1] for f in row['diff']]
            else:
                total_additions = sum(d.diff.count(b'+') for d in row['diff'])
                total_deletions = sum(d.diff.count(b'-') for d in row['diff'])
                file_types = [os.path.splitext(str(d.a_path))[1] for d in row['diff']]

            results.append({
                'original_message': row['message'],
                'generated_message': generated_msg,
                'similarity_score': similarity,
                'author': row['author'],
                'date': row['date'],
                'files_changed': files_modified,
                'additions': total_additions,
                'deletions': total_deletions,
                'file_types': list(set(ft for ft in file_types if ft)),
                'diff_summary': diff_summary[:500],
                'url': row['url']
            })
        
        return pd.DataFrame(results)
    
    def save_results(self, results, base_filename='results'):
        """Save analysis results to both JSON and pickle formats"""
        # Save as JSON (for human readability)
        json_filename = f"{base_filename}.json"
        results_dict = results.to_dict(orient='records')
        for record in results_dict:
            if isinstance(record.get('date'), datetime):
                record['date'] = record['date'].isoformat()
        
        with open(json_filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as pickle (for complete object preservation)
        pickle_filename = f"{base_filename}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {json_filename} and {pickle_filename}")
    
    @staticmethod
    def load_results(base_filename='results'):
        """Load analysis results from pickle file"""
        pickle_filename = f"{base_filename}.pkl"
        try:
            with open(pickle_filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"No saved results found at {pickle_filename}")
            return None
    
    def run_research(self, start_date=None, end_date=None):
        """Run the complete research process"""
        print("Setting up research environment...")
        self.setup()
        
        print("Extracting commits...")
        commits_df = self.extract_commits(start_date, end_date)
        print(f"Found {len(commits_df)} commits")
        
        print("Analyzing commits...")
        analysis_results = self.analyze_messages(commits_df)
        
        print("Saving results...")
        self.save_results(analysis_results)
        
        return analysis_results

    @staticmethod
    def analyze_saved_results(base_filename='results'):
        """Analyze previously saved results with focus on commit message quality"""
        # Load results
        with open(f"{base_filename}.pkl", 'rb') as f:
            results = pickle.load(f)
        
        print("\nCommit Message Analysis:")
        print(f"Total commits analyzed: {len(results)}")
        
        # Message similarity analysis
        print("\nMessage Similarity Analysis:")
        similarity_scores = results['similarity_score']
        print(f"Average similarity score: {similarity_scores.mean():.2f}")
        print(f"Median similarity score: {similarity_scores.median():.2f}")
        print(f"Max similarity score: {similarity_scores.max():.2f}")
        print(f"Min similarity score: {similarity_scores.min():.2f}")
        
        # Code change correlation
        print("\nCode Change Analysis:")
        print(f"Average files changed per commit: {results['files_changed'].mean():.2f}")
        print(f"Average additions per commit: {results['additions'].mean():.2f}")
        print(f"Average deletions per commit: {results['deletions'].mean():.2f}")
        
        # File type analysis
        all_file_types = [ft for fts in results['file_types'] for ft in fts]
        file_type_counts = pd.Series(all_file_types).value_counts()
        print("\nMost Common File Types Modified:")
        for ft, count in file_type_counts.head().items():
            print(f"- {ft or 'no extension'}: {count} files")
        
        # Detailed commit analysis
        print("\nDetailed Commit Analysis:")
        for idx, row in results.iterrows():
            print(f"\nCommit {idx + 1}:")
            print(f"URL: {row['url']}")
            print(f"Author: {row['author']}")
            print(f"Date: {row['date']}")
            print(f"Files changed: {row['files_changed']}")
            print(f"Additions: {row['additions']}")
            print(f"Deletions: {row['deletions']}")
            print("\nOriginal message:")
            print(row['original_message'])
            print("\nGenerated message:")
            print(row['generated_message'])
            print(f"\nSimilarity score: {row['similarity_score']:.2f}")
            print("\n" + "="*80)
        
        return results

if __name__ == "__main__":
    # Example usage - API only mode (no repository clone)
    research = CommitMessageResearch(
        "https://github.com/tensorflow/tensorflow.git",
        use_local_repo=False  # Set to True if you need detailed diffs
    )
    
    # Use a more recent date range to get some commits
    results = research.run_research(
        start_date="2024-03-29",  # Very recent date range
        end_date="2024-03-31"
    )
    
    # Load and analyze saved results
    research = CommitMessageResearch(
        "https://github.com/tensorflow/tensorflow.git",
        use_local_repo=False
    )
    results = research.analyze_saved_results() 