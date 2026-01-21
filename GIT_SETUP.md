# Git Setup Guide

This guide explains how to set up this project as a git repository and use it across multiple computers.

## Initial Setup (First Time)

### 1. Initialize Git Repository

```bash
cd /Users/ajbailey4@ad.wisc.edu/ssms

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Phase 1 complete - Mamba infrastructure setup"
```

### 2. Set Up ROME as a Git Submodule (Recommended)

Since this project depends on the ROME repository, you can add it as a git submodule:

```bash
# If rome/ directory doesn't exist yet
git submodule add https://github.com/kmeng01/rome.git rome

# If rome/ already exists, remove it first
rm -rf rome
git submodule add https://github.com/kmeng01/rome.git rome

# Initialize and update submodule
git submodule init
git submodule update
```

### 3. Create Remote Repository

**Option A: GitHub**
```bash
# Create a new repository on GitHub (via web interface)
# Then connect it:
git remote add origin https://github.com/YOUR_USERNAME/mamba-causal-analysis.git
git branch -M main
git push -u origin main
```

**Option B: GitLab**
```bash
# Create a new repository on GitLab (via web interface)
# Then connect it:
git remote add origin https://gitlab.com/YOUR_USERNAME/mamba-causal-analysis.git
git branch -M main
git push -u origin main
```

**Option C: Private Git Server**
```bash
git remote add origin YOUR_SERVER_URL
git push -u origin main
```

## Cloning on Another Computer

### 1. Clone the Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/mamba-causal-analysis.git

# Or if you forget --recursive
git clone https://github.com/YOUR_USERNAME/mamba-causal-analysis.git
cd mamba-causal-analysis
git submodule init
git submodule update
```

### 2. Set Up Environment

```bash
cd mamba-causal-analysis

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate mamba-causal

# Verify setup
python test_phase1.py
```

## Daily Workflow

### Making Changes

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push
```

### Syncing Across Computers

**On Computer A (after making changes):**
```bash
git add .
git commit -m "Updated causal tracing implementation"
git push
```

**On Computer B (to get changes):**
```bash
git pull
```

### Update Submodules

If the ROME repository is updated:
```bash
git submodule update --remote rome
git commit -am "Updated ROME submodule"
git push
```

## Best Practices

### 1. Branch Strategy

For development work, use branches:

```bash
# Create a new branch for Phase 2
git checkout -b phase2-causal-tracing

# Work on your changes
git add .
git commit -m "Implement basic causal tracing"

# Push branch to remote
git push -u origin phase2-causal-tracing

# When ready, merge to main
git checkout main
git merge phase2-causal-tracing
git push
```

### 2. What NOT to Commit

The `.gitignore` file already excludes:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments
- Downloaded model weights
- Experimental results and outputs
- Jupyter notebook checkpoints
- IDE-specific files

### 3. Large Files

**Don't commit:**
- Model weights (`.pt`, `.pth`, `.safetensors`)
- Large result files (`.npz`, `.pkl`)
- Downloaded datasets

**Do commit:**
- Source code (`.py`)
- Configuration files (`.json`, `.yml`)
- Documentation (`.md`)
- Test scripts
- Requirements files

### 4. Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add mamba_repr_tools.py for extracting token representations"
git commit -m "Fix layer naming bug in ssm_nethook.py"
git commit -m "Update documentation for Phase 2"

# Bad
git commit -m "updates"
git commit -m "fix"
git commit -m "wip"
```

## Useful Git Commands

### Check Status
```bash
git status                    # See what's changed
git log --oneline            # View commit history
git diff                     # See unstaged changes
git diff --staged            # See staged changes
```

### Undo Changes
```bash
git restore <file>           # Discard changes to a file
git restore --staged <file>  # Unstage a file
git reset --soft HEAD~1      # Undo last commit, keep changes
git reset --hard HEAD~1      # Undo last commit, discard changes (DANGEROUS!)
```

### Branches
```bash
git branch                   # List branches
git branch <name>            # Create new branch
git checkout <name>          # Switch to branch
git checkout -b <name>       # Create and switch to branch
git merge <name>             # Merge branch into current
git branch -d <name>         # Delete branch
```

### Remote
```bash
git remote -v                # Show remote URLs
git fetch                    # Download changes without merging
git pull                     # Fetch and merge changes
git push                     # Upload changes
git push -u origin <branch>  # Push new branch and set upstream
```

## Common Scenarios

### Scenario 1: Work on Computer A, Continue on Computer B

**Computer A:**
```bash
git add .
git commit -m "Completed representation extraction"
git push
```

**Computer B:**
```bash
git pull
# Now you have the latest changes
```

### Scenario 2: Made Changes on Both Computers

If you accidentally worked on both without syncing:

**Computer B:**
```bash
git pull  # This will fail if there are conflicts

# If conflicts occur
git status  # Shows conflicting files
# Edit files to resolve conflicts
git add .
git commit -m "Resolved merge conflicts"
git push
```

### Scenario 3: Experiment Branch

For experimental features:

```bash
# Create experiment branch
git checkout -b experiment-ssm-states

# Make experimental changes
# ... work work work ...

# If successful, merge to main
git checkout main
git merge experiment-ssm-states

# If failed, just delete branch
git checkout main
git branch -d experiment-ssm-states
```

## GitHub-Specific Features

### Issues and Tracking

Use GitHub Issues to track tasks:
- Create issues for Phase 2 tasks
- Reference issues in commits: `git commit -m "Fix #5: Layer naming bug"`
- Close issues automatically: `git commit -m "Closes #5: Implement causal tracing"`

### Pull Requests

For collaborative work:
1. Fork the repository
2. Create a branch
3. Make changes
4. Push to your fork
5. Create a Pull Request

### GitHub Actions (Optional)

Create `.github/workflows/tests.yml` to run tests automatically:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
        with:
          environment-file: environment.yml
      - name: Run tests
        run: python test_phase1.py
```

## Backup Strategy

### Regular Backups

```bash
# Push daily
git push

# Create tags for milestones
git tag -a v1.0-phase1 -m "Phase 1 complete"
git push origin v1.0-phase1
```

### Multiple Remotes

Add a backup remote:

```bash
git remote add backup YOUR_BACKUP_URL
git push backup main
```

## Migration Guide

### Moving to a New Computer

1. **Clone repository:**
   ```bash
   git clone --recursive YOUR_REPO_URL
   cd mamba-causal-analysis
   ```

2. **Set up environment:**
   ```bash
   conda env create -f environment.yml
   conda activate mamba-causal
   ```

3. **Verify setup:**
   ```bash
   python test_phase1.py
   ```

4. **Start working:**
   ```bash
   # You're ready to go!
   ```

## Troubleshooting

### Problem: Submodule not cloned

```bash
git submodule init
git submodule update
```

### Problem: Large files accidentally committed

```bash
# Use git filter-branch or BFG Repo-Cleaner
# See: https://docs.github.com/en/repositories/working-with-files/managing-large-files
```

### Problem: Merge conflicts

```bash
git status  # See conflicting files
# Edit files manually
git add .
git commit -m "Resolved conflicts"
```

### Problem: Wrong files committed

```bash
# Before pushing
git reset --soft HEAD~1  # Undo commit, keep changes
# Fix .gitignore
git add .
git commit -m "Fixed commit"

# After pushing (more complex)
# Contact repository admin
```

## Summary

**Essential commands:**
```bash
# Daily workflow
git pull          # Get latest changes
git add .         # Stage changes
git commit -m ""  # Commit changes
git push          # Upload changes

# Setup on new computer
git clone --recursive <url>
conda env create -f environment.yml
python test_phase1.py
```

**Remember:**
- Commit often
- Push regularly
- Pull before starting work
- Use descriptive commit messages
- Don't commit large files or model weights
