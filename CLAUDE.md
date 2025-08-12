# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal study repository focused on AI research and development. It contains academic paper implementations, AI projects, and development tools organized for learning and experimentation.

## Key Architecture

### Main Directory Structure
- **`archive/`**: Historical study materials and implementations
  - `study_AI/`: PyTorch implementations of classic AI papers and experiments
  - `study_cpp/`: C++ learning projects
  - `etc/`: Utility scripts for development tasks
- **`projects/`**: Active development projects
  - `Timer_App/`: Standalone Pomodoro timer application (see its CLAUDE.md)
- **`Papers/`**: Academic papers being studied with tracking list
- **`log/`**: Study session logs by date

### Technology Stack
- **Python**: Primary language for AI implementations using PyTorch
- **C++**: Secondary language for systems programming study
- **SQLite**: Database for project data storage
- **No package management**: Individual Python scripts without requirements.txt or virtual environments

## Development Patterns

### AI Study Files
- Paper implementations follow naming convention: `YYYY_papername.py` (e.g., `2012_alexnet.py`)
- Each implementation is self-contained with PyTorch neural network definitions
- Files include Korean comments for learning purposes
- Common pattern: Class-based model definitions inheriting from `nn.Module`

### Project Structure
- Active projects maintain their own CLAUDE.md files with specific instructions
- Database files (`.db`) are created automatically and should not be version controlled
- Build artifacts (`build/`, `dist/`, `*.spec`) are temporary and can be cleaned

### Logging
- Study sessions are manually logged in `log/log_YYMMDD.txt` format
- Papers being studied are tracked in `Papers/papers.txt`

## Common Commands

### Python AI Development
```bash
# Run individual study scripts
python archive/study_AI/2012_alexnet.py

# Check CUDA availability
python archive/etc/cuda_check.py
```

### Project Development
```bash
# Navigate to specific project
cd projects/Timer_App

# Follow project-specific CLAUDE.md instructions
```

### Utility Scripts
- `cuda_check.py`: Verify PyTorch CUDA installation
- `kaggle_download.py`: Download datasets from Kaggle

## File Patterns to Recognize

- **Study implementations**: Located in `archive/study_AI/` with descriptive names
- **Active projects**: Each has its own directory under `projects/` with dedicated documentation
- **Temporary files**: `*.db`, `build/`, `dist/` directories are auto-generated
- **Learning logs**: Date-stamped text files in `log/` directory for progress tracking