# TODO: Add movement contrains by resetting the y-coordinate of the mouse
# to the slider. That way the mouse device can be moved freely and accidental
# click won't cause any problems.
# TODO: Find a way to represent pixel coordinates of the slider.

import subprocess
import sys

def install(package):
    """Installs the specified package using pip in a subprocess"""
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

# Attempts to import the 'mouse' module, and installs it if necessary
try:
    import mouse
except ImportError:
    try:
        install("mouse")
        import mouse
    except Exception as e:
        print(f"Failed to install and import 'mouse': {e}")

def hold_mouse():
    """Holds down the left mouse button"""
    mouse.hold(button="left")

def check_mouse():
    """Checks if the left mouse button is being held down"""
    if not mouse.is_pressed(button='left'):
        mouse.hold(button='left')

def release_mouse():
    """Releases the left mouse button"""
    mouse.release(button="left")

if __name__ == "__main__":
    hold_mouse() # trolling