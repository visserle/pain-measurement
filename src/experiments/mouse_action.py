# work in progress

# TODO:
# - add support for multiple monitors, maybe using the screeninfo library? and how does mouse behave?
# -> good news, y-coordinates are the same for all monitors, so we don't need to worry about that
# -> although we need to find out from which monitor the y-coordinate is taken in psychopy TODO
# -> should be window-dependent
# from win32api import GetSystemMetrics
# width = user32.GetSystemMetrics(0)
# height = user32.GetSystemMetrics(1)
# # in a multi-monitor setup, you can use the screeninfo library, for instance
# # >>> from screeninfo import get_monitors
# # >>> for m in get_monitors():
# # >>>    print(str(m))
# # this prints out the resolution (width and height), position (x, y), and 
# other information for each monitor 
# print(width, height)

# https://www.phind.com/search?cache=b7ba5b4f-e9d5-4e1f-8bb1-bed231f1c5a9
# How can I use the python `mouse` (https://pypi.org/project/mouse/) package with multiple monitors?


"""
This script uses the 'mouse' library to manipulate the mouse state.

The 'mouse' library is written in pure Python library with no dependencies and provides functions to simulate mouse interactions.
It works on both Windows and Linux. More details about the library can be found here: https://pypi.org/project/mouse/

This script also adjusts the DPI awareness of the application using ctypes to ensure that the correct screen resolution is retrieved 
even when DPI scaling is in use. This method is safe to use with all types of monitors, whether they use DPI scaling or not.
"""

import ctypes
from .psychopy_utils import psychopy_import
mouse = psychopy_import("mouse")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()


def hold():
    """
    Simulates holding down the left mouse button.
    """
    mouse.hold(button="left")


def check(pixel_y):
    """
    Checks if the mouse is pressed and moves it to the pixel y-coordinate if necessary.

    This function can be used in combination with a psychopy slider component for continuous ratings by calling it every frame.

    Parameters
    ----------
    pixel_y : float
        The y-coordinate in pixel to which the mouse should be moved if it is not already there.

    Notes
    -----
    If the mouse is not currently being held down (as indicated by mouse.is_pressed), 
    this function will hold it down using the `hold` function. 
    It then checks if the current y-coordinate of the mouse (as indicated by mouse.get_position) 
    is within a certain range of the desired y-coordinate. 
    If it is not, it moves the mouse to the desired y-coordinate using mouse.move.

    """
    if not mouse.is_pressed(button='left'):
        hold()
    if not (pixel_y*0.9 < mouse.get_position()[1] < pixel_y*1.1): # get y-coordinate
        mouse.move(mouse.get_position()[0], pixel_y, absolute=True, duration=0) # move to slider position

def release():
    """
    Simulates the release of the left mouse button.
    """
    mouse.release(button="left")


def pixel_pos_y(component_pos, win_size, win_pos):
    """
    Converts PsychoPy coordinates of a component to the pixel y-coordinate.
    
    The function is used to convert the y-coordinate from the PsychoPy coordinate system 
    (where the y-axis goes upwards and the coordinates are normalized to the range [-1, 1])
    to the pixel coordinate system (where the y-axis goes downwards and the coordinates are
    in the range [0, height]).

    Note that this function needs DPI awareness to work correctly.
    
    Parameters
    ----------
    component_pos : Tuple[float, float]
        PsychoPy coordinates of the component (x, y). 
        For instance, see the PsychoPy Slider Component documentation:
        https://www.psychopy.org/builder/components/slider.html#properties

    win_size : Tuple[int, int]
        Window size in pixels (width, height). 

    win_pos : Tuple[int, int]
        Window position in pixels (x, y). This is the position of the top-left corner of the window
        relative to the screen. 

    Returns
    -------
    float
        The y-coordinate in the pixel coordinate system. This is the distance from the top of the screen.

    Example for PsychoPy Slider Component
    --------
    >>> pixel_pos_y(slider.pos, win.size, win.pos)
    450.0
    """
    _, y = component_pos
    _, height = win_size
    _, win_y = win_pos

    # To convert the y-coordinate, we subtract it from 1 to flip it 
    # (because PsychoPy's y-axis is upwards while pixel count downwards), 
    # divide by 2 to scale it to [0, 1], and multiply by `height` to scale it to [0, height]. 
    # Then we add `win_y` to account for the window's position.
    pixel_y = (1 - y) / 2 * height + win_y
    return pixel_y


if __name__ == "__main__":
    hold() # trolling
