# work in progress

# TODO:
# - Add movement contrains by resetting the y-coordinate of the mouse
#   to the slider. That way the mouse device can be moved freely and accidental
#   clicks won't cause any problems.
# - Find a way to represent pixel coordinates of the slider in psychopy
# - maybe add check if we are in a psychopy experiment? (very useful for aborting experiments)

import subprocess
import sys

def install(package):
    """Installs the specified package using pip in a subprocess"""
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

# Attempts to import the 'mouse' package, and installs it if necessary
# The mouse package is in pure python (no dependencies) and works on Windows and Linux.
# https://pypi.org/project/mouse/

try:
    import mouse
except ImportError:
    try:
        install("mouse")
        import mouse
    except Exception as e:
        print(f"Failed to install and import 'mouse': {e}")

def hold():
    """Holds down the left mouse button"""
    mouse.hold(button="left")

def check():
    """Checks if the left mouse button is being held down and holds it if not"""
    if not mouse.is_pressed(button='left'):
        hold()

def release():
    """Releases the left mouse button"""
    mouse.release(button="left")

if __name__ == "__main__":
    hold() # trollin'