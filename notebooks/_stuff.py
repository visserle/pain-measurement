import tkinter as tk
from pynput.mouse import Listener

def on_click(x, y, button, pressed):
    if button.name == 'left' and pressed:
        # Display the animation at the cursor's position
        display_animation(x, y)

def display_animation(x, y):
    # Create a new Tkinter window
    anim_window = tk.Tk()
    anim_window.overrideredirect(True)  # Hide the title bar
    anim_window.geometry(f"+{x}+{y}")  # Position the window

    # Create a Canvas and add an oval (example animation)
    canvas = tk.Canvas(anim_window, width=100, height=100)
    canvas.pack()
    canvas.create_oval(25, 25, 75, 75, fill='red')

    # Close the window after a short duration
    anim_window.after(500, anim_window.destroy)  # Adjust the duration as needed
    anim_window.mainloop()

# Start listening for mouse clicks
listener = Listener(on_click=on_click)
listener.start()
listener.join()
