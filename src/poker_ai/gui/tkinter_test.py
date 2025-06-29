import tkinter as tk
try:
    print(f"Tkinter version: {tk.Tcl().eval('info patchlevel')}")
    root = tk.Tk()
    label = tk.Label(root, text="Tkinter is working!")
    label.pack()
    print("Tkinter test script: Root window created, label packed.")
    # root.mainloop() # Not running mainloop for a quick test
    root.destroy() # Clean up
    print("Tkinter test script: Root window destroyed.")
except Exception as e:
    print(f"Tkinter test script: Error: {e}")
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    import os
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print(f"sys.path: {sys.path}")
