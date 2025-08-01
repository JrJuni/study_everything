import tkinter as tk
from tkinter import messagebox, ttk
import sqlite3
import datetime
import winsound
import sys

DB_PATH = 'tasks.db'

class PomodoroTimer:
    def __init__(self, master):
        self.master = master
        self.master.title("Customizable Pomodoro Timer")
        self.master.resizable(False, False)
        self.master.geometry("450x350")

        # Initialize database for work session tasks
        self.conn = sqlite3.connect(DB_PATH)
        self._create_tasks_table()

        # Default settings variables
        self.work_var = tk.StringVar(value="25")
        self.short_break_var = tk.StringVar(value="5")
        self.long_break_var = tk.StringVar(value="15")
        self.cycles_var = tk.StringVar(value="4")
        self.rounds_var = tk.StringVar(value="1")

        # Apply initial settings and initialize sessions
        self._apply_settings(rebuild_tasks=True)

        # Menu bar with Settings popup
        menubar = tk.Menu(master)
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Time", command=self.open_settings)
        settings_menu.add_command(label="Tasks", command=self.open_tasks)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        master.config(menu=menubar)

        # Summary label (3 lines)
        self.summary_label = tk.Label(
            master,
            text=self._generate_summary(),
            font=("Helvetica", 12, "italic"),
            justify="center"
        )
        self.summary_label.pack(pady=(10, 5))

        # Pomodoro UI state
        self.mode = 'Work'
        self.remaining = self.work_duration
        self.is_running = False
        self.current_cycle = 0
        self.current_round = 0
        self.session_count = 0
        self.tick_job = None  # Track the scheduled tick job

        # Mode and timer labels
        self.mode_label = tk.Label(master, text="Work", font=("Helvetica", 14, "bold"))
        self.mode_label.pack(pady=(5, 0))
        self.timer_label = tk.Label(master, text=self._format_time(self.remaining), font=("Helvetica", 36))
        self.timer_label.pack(pady=10)

        # Controls: Start, Pause, Reset
        controls = tk.Frame(master)
        controls.pack(pady=5)
        self.start_button = tk.Button(controls, text="Start", width=10, bg="#4CAF50", fg="white", command=self.start)
        self.start_button.grid(row=0, column=0, padx=5)
        self.pause_button = tk.Button(controls, text="Pause", width=10, state="disabled", command=self.pause)
        self.pause_button.grid(row=0, column=1, padx=5)
        self.reset_button = tk.Button(controls, text="Reset", width=10, bg="#F44336", fg="white", state="disabled", command=self.reset)
        self.reset_button.grid(row=0, column=2, padx=5)
        
        # Bind spacebar for start/pause toggle
        self.master.bind("<KeyPress-space>", self.toggle_timer)
        self.master.focus_set()  # Ensure main window can receive key events

    # Start timer
    def start(self):
        if not self.is_running:
            if not self._apply_settings(): return
            # Cancel any existing tick job before starting
            if self.tick_job:
                self.master.after_cancel(self.tick_job)
                self.tick_job = None
            self.is_running = True
            self.start_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.reset_button.config(state='disabled')
            self._update_mode_label()
            self._tick()

    # Pause timer
    def pause(self):
        if self.is_running:
            self.is_running = False
            # Cancel the scheduled tick job
            if self.tick_job:
                self.master.after_cancel(self.tick_job)
                self.tick_job = None
            self.start_button.config(state='normal')
            self.pause_button.config(state='disabled')
            self.reset_button.config(state='normal')

    # Reset timer
    def reset(self):
        self.is_running = False
        # Cancel any scheduled tick job
        if self.tick_job:
            self.master.after_cancel(self.tick_job)
            self.tick_job = None
        self.current_cycle = 0
        self.current_round = 0
        self.session_count = 0
        self.mode = 'Work'
        self.remaining = self.work_duration
        self.mode_label.config(text="Work")
        self.timer_label.config(text=self._format_time(self.remaining))
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled')
        self.reset_button.config(state='disabled')

    # Toggle timer with spacebar
    def toggle_timer(self, event=None):
        if not self.is_running:
            # If not running, start the timer
            self.start()
        else:
            # If running, pause the timer
            self.pause()
        return "break"  # Prevent default spacebar behavior

    # Apply settings and rebuild sessions
    def _apply_settings(self, rebuild_tasks=False):
        try:
            w = int(self.work_var.get()); sb = int(self.short_break_var.get())
            lb = int(self.long_break_var.get()); sc = int(self.cycles_var.get()); rc = int(self.rounds_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "All settings must be integers.")
            return False
        self.work_duration = w * 60
        self.short_break_duration = sb * 60
        self.long_break_duration = lb * 60
        self.cycles_per_round = sc
        self.rounds_total = rc
        
        # Only initialize sessions when explicitly requested (settings change)
        if rebuild_tasks:
            c = self.conn.cursor()
            c.execute("DELETE FROM WorkTasks")
            total = sc * rc
            for i in range(1, total+1):
                # Create default task names: "Task 01", "Task 02", etc.
                default_task = f"Task {i:02d}"
                c.execute("INSERT INTO WorkTasks(session, task) VALUES(?,?)", (i, default_task))
            self.conn.commit()
        else:
            # Ensure we have enough tasks for current settings
            c = self.conn.cursor()
            c.execute("SELECT COUNT(*) FROM WorkTasks")
            current_count = c.fetchone()[0]
            total_needed = sc * rc
            if current_count < total_needed:
                # Add missing tasks without deleting existing ones
                for i in range(current_count + 1, total_needed + 1):
                    default_task = f"Task {i:02d}"
                    c.execute("INSERT INTO WorkTasks(session, task) VALUES(?,?)", (i, default_task))
                self.conn.commit()
        
        if hasattr(self, 'summary_label'):
            self.summary_label.config(text=self._generate_summary())
        return True

    # Timer tick
    def _tick(self):
        if self.is_running and self.remaining > 0:
            self.remaining -= 1
            self.timer_label.config(text=self._format_time(self.remaining))
            self.tick_job = self.master.after(1000, self._tick)
        elif self.is_running:
            self._cycle_complete()

    # Switch modes
    def _cycle_complete(self):
        # Play completion sound and show notification
        self._play_completion_sound()
        
        if self.mode == 'Work':
            self.current_cycle += 1
            self.session_count += 1
            next_mode = 'Short Break' if self.current_cycle < self.cycles_per_round else 'Long Break'
            
            # Show work completion message
            task_name = self._current_task()
            task_display = f" - {task_name}" if task_name else ""
            messagebox.showinfo("Work Complete!", 
                               f"Work session {self.session_count} completed{task_display}!\n\n"
                               f"Time for a {next_mode.lower()}.")
            self.mode = next_mode
        else:
            break_type = self.mode.lower()
            if self.mode == 'Long Break':
                self.current_round += 1
                self.current_cycle = 0
                if self.current_round >= self.rounds_total:
                    messagebox.showinfo("Pomodoro Timer", "🎉 All rounds completed! Great job!")
                    self.reset()
                    return
                else:
                    messagebox.showinfo("Break Complete!", 
                                       f"{self.mode} completed!\n\n"
                                       f"Ready to start Round {self.current_round + 1}?")
            else:
                messagebox.showinfo("Break Complete!", 
                                   f"{self.mode} completed!\n\n"
                                   f"Ready to get back to work?")
            self.mode = 'Work'
            
        self.remaining = {
            'Work': self.work_duration,
            'Short Break': self.short_break_duration,
            'Long Break': self.long_break_duration
        }[self.mode]
        self._update_mode_label()
        self.timer_label.config(text=self._format_time(self.remaining))
        self.tick_job = self.master.after(1000, self._tick)

    # Update mode label
    def _update_mode_label(self):
        if self.mode == 'Work':
            task = self._current_task()
            self.mode_label.config(text=f"Work : {task}")
        else:
            self.mode_label.config(text=self.mode)

    # Get current task
    def _current_task(self):
        c = self.conn.cursor()
        # session_count is incremented AFTER work completion, so current session is session_count + 1
        current_session = self.session_count + 1
        c.execute("SELECT task FROM WorkTasks WHERE session=?", (current_session,))
        row = c.fetchone()
        return row[0] if row else ''

    # Settings popup
    def open_settings(self):
        dlg = tk.Toplevel(self.master)
        dlg.title("Pomodoro Settings")
        dlg.grab_set()
        fields = [
            ("Work (minutes)", self.work_var),
            ("Short Break (minutes)", self.short_break_var),
            ("Long Break (minutes)", self.long_break_var),
            ("Cycles per round", self.cycles_var),
            ("Full rounds", self.rounds_var)
        ]
        for i,(lbl,var) in enumerate(fields):
            tk.Label(dlg, text=lbl+":").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(dlg, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=2)
        def on_ok():
            if self._apply_settings(rebuild_tasks=True):
                self.reset()
                dlg.destroy()
        btns = tk.Frame(dlg);
        btns.grid(row=len(fields), columnspan=2, pady=10)
        tk.Button(btns, text="OK", width=8, command=on_ok).pack(side="left", padx=5)
        tk.Button(btns, text="Cancel", width=8, command=dlg.destroy).pack(side="right", padx=5)

    # Tasks popup with editable table
    def open_tasks(self):
        dlg = tk.Toplevel(self.master)
        dlg.title("Manage Tasks")
        dlg.grab_set()
        dlg.geometry("500x400")
        
        # Track active editing entry
        active_entry = {"entry": None, "item": None, "session": None}
        
        # Create treeview for editable table
        columns = ("Session", "Task")
        tree = ttk.Treeview(dlg, columns=columns, show="headings", height=15)
        tree.heading("Session", text="Session #")
        tree.heading("Task", text="Task Description")
        tree.column("Session", width=80, anchor="center")
        tree.column("Task", width=350)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(dlg, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # Load current tasks into tree
        def load_tasks():
            for item in tree.get_children():
                tree.delete(item)
            c = self.conn.cursor()
            for session, task in c.execute("SELECT session, task FROM WorkTasks ORDER BY session"):
                tree.insert("", "end", values=(session, task))
        
        # Save current edit if any
        def save_current_edit():
            if active_entry["entry"] and active_entry["entry"].winfo_exists():
                try:
                    new_task = active_entry["entry"].get().strip()
                    c = self.conn.cursor()
                    c.execute("UPDATE WorkTasks SET task=? WHERE session=?", (new_task, active_entry["session"]))
                    self.conn.commit()
                    tree.item(active_entry["item"], values=(active_entry["session"], new_task))
                finally:
                    if active_entry["entry"]:
                        active_entry["entry"].destroy()
                    active_entry["entry"] = None
                    active_entry["item"] = None
                    active_entry["session"] = None
        
        # Handle single-click for editing
        def on_single_click(event):
            # Save any existing edit first
            if active_entry["entry"]:
                save_current_edit()
            
            item = tree.selection()
            if not item:
                return
            item = item[0]
            column = tree.identify_column(event.x)
            
            if column == "#2":  # Only allow editing task column
                # Get current values
                values = tree.item(item, "values")
                session_num, current_task = values
                
                # Create edit entry
                bbox = tree.bbox(item, column)
                if bbox:
                    entry = tk.Entry(dlg)
                    entry.place(x=bbox[0] + tree.winfo_x(), 
                              y=bbox[1] + tree.winfo_y(), 
                              width=bbox[2], height=bbox[3])
                    entry.insert(0, current_task)
                    entry.focus()
                    entry.select_range(0, tk.END)
                    
                    # Update active entry tracking
                    active_entry["entry"] = entry
                    active_entry["item"] = item
                    active_entry["session"] = session_num
                    
                    def save_edit(event=None):
                        save_current_edit()
                    
                    def cancel_edit(event=None):
                        if active_entry["entry"]:
                            active_entry["entry"].destroy()
                            active_entry["entry"] = None
                            active_entry["item"] = None
                            active_entry["session"] = None
                    
                    entry.bind("<Return>", save_edit)
                    entry.bind("<Escape>", cancel_edit)
                    entry.bind("<FocusOut>", save_edit)
        
        tree.bind("<Button-1>", on_single_click)
        
        # Handle keyboard input for direct editing and navigation
        def on_key_press(event):
            # Handle arrow keys for navigation when no entry is active
            if not active_entry["entry"]:
                if event.keysym in ['Up', 'Down', 'Left', 'Right']:
                    return  # Let tree handle navigation naturally
                elif event.keysym == 'Return':
                    # Enter key starts editing the selected item
                    selection = tree.selection()
                    if selection:
                        item = selection[0]
                        values = tree.item(item, "values")
                        session_num, current_task = values
                        
                        bbox = tree.bbox(item, "#2")  # Task column
                        if bbox:
                            entry = tk.Entry(dlg)
                            entry.place(x=bbox[0] + tree.winfo_x(), 
                                      y=bbox[1] + tree.winfo_y(), 
                                      width=bbox[2], height=bbox[3])
                            entry.insert(0, current_task)  # Start with current content
                            entry.focus()
                            entry.select_range(0, tk.END)  # Select all text
                            
                            # Update active entry tracking
                            active_entry["entry"] = entry
                            active_entry["item"] = item
                            active_entry["session"] = session_num
                            
                            def save_edit_and_next(event=None):
                                save_current_edit()
                                # Move to next item after saving
                                try:
                                    children = tree.get_children()
                                    current_index = children.index(item)
                                    if current_index < len(children) - 1:
                                        next_item = children[current_index + 1]
                                        tree.selection_set(next_item)
                                        tree.focus(next_item)
                                        tree.see(next_item)
                                    # If this was the last item, just stay on current item
                                except:
                                    pass
                                # Always ensure tree regains focus for keyboard events
                                tree.focus_set()
                                dlg.focus_set()
                            
                            def cancel_edit(event=None):
                                if active_entry["entry"]:
                                    active_entry["entry"].destroy()
                                    active_entry["entry"] = None
                                    active_entry["item"] = None
                                    active_entry["session"] = None
                                # Return focus to tree and ensure it can receive events
                                tree.focus_set()
                                dlg.focus_set()
                            
                            entry.bind("<Return>", save_edit_and_next)
                            entry.bind("<Escape>", cancel_edit)
                            entry.bind("<FocusOut>", lambda e: save_current_edit())
                    return "break"
                elif event.char and event.char.isprintable():
                    # Start editing with the typed character
                    selection = tree.selection()
                    if selection:
                        item = selection[0]
                        values = tree.item(item, "values")
                        session_num, current_task = values
                        
                        bbox = tree.bbox(item, "#2")  # Task column
                        if bbox:
                            entry = tk.Entry(dlg)
                            entry.place(x=bbox[0] + tree.winfo_x(), 
                                      y=bbox[1] + tree.winfo_y(), 
                                      width=bbox[2], height=bbox[3])
                            entry.insert(0, event.char)  # Start with typed character
                            entry.focus()
                            entry.icursor(tk.END)
                            
                            # Update active entry tracking
                            active_entry["entry"] = entry
                            active_entry["item"] = item
                            active_entry["session"] = session_num
                            
                            def save_edit_and_next(event=None):
                                save_current_edit()
                                # Move to next item after saving
                                try:
                                    children = tree.get_children()
                                    current_index = children.index(item)
                                    if current_index < len(children) - 1:
                                        next_item = children[current_index + 1]
                                        tree.selection_set(next_item)
                                        tree.focus(next_item)
                                        tree.see(next_item)
                                    # If this was the last item, just stay on current item
                                except:
                                    pass
                                # Always ensure tree regains focus for keyboard events
                                tree.focus_set()
                                dlg.focus_set()
                            
                            def cancel_edit(event=None):
                                if active_entry["entry"]:
                                    active_entry["entry"].destroy()
                                    active_entry["entry"] = None
                                    active_entry["item"] = None
                                    active_entry["session"] = None
                                # Return focus to tree and ensure it can receive events
                                tree.focus_set()
                                dlg.focus_set()
                            
                            entry.bind("<Return>", save_edit_and_next)
                            entry.bind("<Escape>", cancel_edit)
                            entry.bind("<FocusOut>", lambda e: save_current_edit())
                        return "break"
        
        # Bind key events
        dlg.bind("<KeyPress>", on_key_press)
        tree.bind("<KeyPress>", on_key_press)
        
        # Enable tree navigation with focus maintenance
        def maintain_focus_after_nav(event):
            # Allow default navigation behavior
            tree.after_idle(lambda: (tree.focus_set(), dlg.focus_set()))
        
        tree.bind("<Up>", maintain_focus_after_nav)
        tree.bind("<Down>", maintain_focus_after_nav)
        tree.bind("<Left>", maintain_focus_after_nav)
        tree.bind("<Right>", maintain_focus_after_nav)
        
        # Handle window close to save any pending edits
        def on_closing():
            save_current_edit()
            dlg.destroy()
        
        # Override window close protocol
        dlg.protocol("WM_DELETE_WINDOW", on_closing)
        
        load_tasks()
        
        # Auto-select first item for immediate typing
        if tree.get_children():
            first_item = tree.get_children()[0]
            tree.selection_set(first_item)
            tree.focus(first_item)
        
        # Ensure tree has focus for keyboard events
        tree.focus_set()

    def _generate_summary(self):
        w,sb = int(self.work_var.get()), int(self.short_break_var.get())
        sc, lb, rc = int(self.cycles_var.get()), int(self.long_break_var.get()), int(self.rounds_var.get())
        ml = lambda x: "min" if x == 1 else "mins"
        cl = lambda x: "Cycle" if x == 1 else "Cycles"
        rl = lambda x: "Round" if x == 1 else "Rounds"
        line1 = f"Work {w} {ml(w)} & Short Break {sb} {ml(sb)} x {sc} {cl(sc)}"
        line2 = f"Long Break {lb} {ml(lb)}"
        line3 = f"Total {rc} {rl(rc)}"
        return f"{line1}\n{line2}\n{line3}"

    def _create_tasks_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS WorkTasks(
                        session INTEGER PRIMARY KEY, task TEXT
                    )''')
        self.conn.commit()

    def _format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

    # Play completion sound (respects system mute)
    def _play_completion_sound(self):
        try:
            if sys.platform == "win32":
                # === WINDOWS SOUND OPTIONS ===
                
                # Option 1: System notification sound (DEFAULT - respects system volume/mute)
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                
                # Option 2: Different system sounds (uncomment to use)
                # winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # Warning sound
                # winsound.MessageBeep(winsound.MB_ICONHAND)         # Error sound
                # winsound.MessageBeep(winsound.MB_ICONQUESTION)     # Question sound
                # winsound.MessageBeep(winsound.MB_OK)               # Default beep
                
                # Option 3: Custom WAV file (uncomment and specify path)
                # winsound.PlaySound("sample.wav", winsound.SND_FILENAME)  # WAV files only
                
                # Option 3b: MP3 and other formats (requires pygame - uncomment to use)
                # Note: Install pygame first: pip install pygame
                # import pygame
                # pygame.mixer.init()
                # pygame.mixer.music.load("sample.mp3")  # Supports MP3, OGG, etc.
                # pygame.mixer.music.play()
                
                # Option 3c: Alternative with playsound library (uncomment to use)
                # Note: Install playsound first: pip install playsound
                # from playsound import playsound
                # playsound("sample.mp3")  # Supports MP3, WAV, etc.
                
                # Option 4: System sounds by name (uncomment to use)
                # winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
                # winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                # winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
                # winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                # winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)
                
            else:
                # For other platforms, use system bell (if available)
                print('\a')  # ASCII bell character
                
        except Exception:
            # If sound fails, continue silently (respects system mute or missing audio)
            pass

if __name__ == "__main__":
    root = tk.Tk()
    PomodoroTimer(root)
    root.mainloop()
