# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Customizable Pomodoro Timer** application built with Python Tkinter. It's a desktop productivity app that helps users manage work sessions using the Pomodoro Technique with customizable timing and task management features.

## Key Architecture

- **Single-file application**: `app.py` contains the entire application in one `PomodoroTimer` class
- **SQLite database**: `tasks.db` stores work session tasks (auto-created on first run)
- **Executable distribution**: Uses PyInstaller to create standalone Windows executables

### Core Components

- **Timer Logic**: Work/break cycle management with sound notifications
- **Settings Management**: Configurable work duration, break times, cycles, and rounds
- **Task Management**: Editable task list with keyboard shortcuts for efficient editing
- **Menu Structure**: "Settings" menu containing "Time" and "Tasks" options

## Development Commands

### Building Distribution
```bash
# Create standalone executable (windowed, no console)
pyinstaller --onefile --windowed --name Timer_App app.py

# Output location: dist/Timer_App.exe
```

### Running Development Version
```bash
python app.py
```

### Clean Build Files
```bash
# Remove build artifacts before rebuilding
rm -rf build/ dist/ *.spec
```

## Database Schema

The application uses SQLite with a single table:
```sql
WorkTasks(session INTEGER PRIMARY KEY, task TEXT)
```

Tasks are automatically generated as "Task 01", "Task 02", etc., based on cycles × rounds configuration.

## Distribution Notes

- **Single-file deployment**: Only `Timer_App.exe` needs to be distributed
- **Database auto-creation**: `tasks.db` is created automatically in the executable's directory
- **No external dependencies**: All Python libraries are bundled in the executable

## UI Interactions

- **Spacebar**: Start/pause timer toggle
- **Task editing**: Single-click or Enter key to edit, Return to save and move to next
- **Sound notifications**: Uses Windows system sounds (respects system mute)

## Key Settings

- Work duration, short/long break times (in minutes)
- Cycles per round (work sessions before long break)
- Total rounds (complete pomodoro sequences)