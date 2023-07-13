# work in progress

# TODO:
# - Add movement contrains by resetting the y-coordinate of the mouse
#   to the slider coordinates in psychopy. That way the mouse device can be moved freely and accidental
#   clicks won't cause any problems in the continous rating.
#   -> For this to work, find a way to represent pixel coordinates of the slider in psychopy
#   -> also maybe add a check if we are in a psychopy experiment? (very useful for aborting experiments)
# - add support for multiple monitors, maybe using the screeninfo library
# width = user32.GetSystemMetrics(0)
# height = user32.GetSystemMetrics(1)
# # in a multi-monitor setup, you can use the screeninfo library, for instance
# # >>> from screeninfo import get_monitors
# # >>> for m in get_monitors():
# # >>>    print(str(m))
# # this prints out the resolution (width and height), position (x, y), and other information for each monitor 
# print(width, height)


"""The mouse library is in pure python (no dependencies) and works on Windows and Linux (https://pypi.org/project/mouse/)
"""
import subprocess
import sys
# needed to get the correct screen resolution when using DPI scaling
# this method is safe to use with all types of monitors, whether they use DPI scaling or not
from win32api import GetSystemMetrics
import ctypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

try:
    import mouse
except ImportError:
    try:
        # install mouse using pip in a subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", mouse], check=True)
    except Exception as e:
        print(f"Failed to install and import '{mouse}': {e}")

def hold():
    """Holds down the left mouse button"""
    mouse.hold(button="left")

def check(pos_y):
    """Checks if the mouse is pressed and moves it to the slider position if necessary"""
    if not mouse.is_pressed(button='left'):
        hold()
    if not mouse.get_position()[1] == pos_y: # get y-coordinate
        mouse.move(mouse.get_position()[0], pos_y * 1.1, absolute=True, duration=0) # move to slider position

def release():
    """Releases the left mouse button"""
    mouse.release(button="left")

def psychopy_to_pixel_coordinates(component_pos, win_size, win_pos):
    """
    Converts PsychoPy coordinates to pixel coordinates
    
    Parameters
    ----------
    component_pos : tuple
        PsychoPy coordinates of the component (x, y).
        For instance, see https://www.psychopy.org/builder/components/slider.html#properties:

        "Position(X,Y): 
            The position of the centre of the stimulus, in the units specified by the stimulus or window. 
            Default is centered left-right, and somewhat lower than the vertical center (0, -0.4)."


    win_size : tuple
        Window size in pixels (width, height)
    win_pos : tuple
        Window position in pixels (x, y)
    
    Returns
    -------
    tuple
        Pixel coordinates of the psychopy component (x, y). 
        The origin (0, 0) is at the top-left corner of the screen.
        Positive x-values are to the right, and positive y-values are downwards.

    Example
    -------
    In the following example, we convert the PsychoPy coordinates of a visual analog scale to pixel coordinates:
    ```python
    psychopy_to_pixel_coordinates(
        pos=vas.pos,
        win_size=win.size,
        win_pos=win.pos)
    ```
    """
    x, y = component_pos
    width, height = win_size
    win_x, win_y = win_pos

    # To convert the x-coordinate, we add 1 to shift it from [-1, 1] to [0, 2], 
    # divide by 2 to scale it to [0, 1], and multiply by `width` to scale it to [0, width].
    # Then we add `win_x` to account for the window's position.
    # To convert the y-coordinate, we subtract it from 1 to flip it 
    # (because PsychoPy's y-axis is upwards while pixel count downwards), 
    # divide by 2 to scale it to [0, 1], and multiply by `height` to scale it to [0, height]. 
    # Then we add `win_y` to account for the window's position.
    pixel_x = (x + 1) / 2 * width + win_x
    pixel_y = (1 - y) / 2 * height + win_y
    return pixel_x, pixel_y


if __name__ == "__main__":
    hold() # trollin'