#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.3),
    on Juli 07, 2023, at 11:36
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from stimuli
from stimuli import StimuliFunction
# Run 'Before Experiment' code from mouse
import subprocess
def install(package):
    """Installs the specified package using pip in a subprocess"""
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

# The following code provides a workaround for using the standard slider 
# component without the need to press any buttons. It relies solely on the
# x-coordinate of the mouse as input for the rating while the mouse is being 
# clicked and held automatically.
# Note: All mouse code is imperative as running functions from the terminal
# returns errors (see https://github.com/boppreh/mouse/issues/34).
# Perhaps pyautogui or pynput could be used within functions, however the current
# imperative mouse implementation just works.

"""Attempts to import the 'mouse' module, and installs it if necessary"""
try:
    import mouse
except ImportError:
    try:
        install("mouse")
        import mouse
    except Exception as e:
        print(f"Failed to install and import 'mouse': {e}")


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.3'
expName = 'mpad1_exp'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'pain_threshold': '38',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\drive\\PhD\\Code\\mpad-pilot\\experiments\\mpad1\\experiment\\mpad1_exp_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[900, 600], fullscr=False, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = True
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "welcome_screen" ---
# Run 'Begin Experiment' code from stimuli
seed = 160
duration = 200 # in seconds
amplitudes = [1, 1.5] # the range will be 2 * sum(amplitudes)
periods = [67, 10]  # 1 : 3 gives a good result
frequencies = 1./np.array(periods)
baseline_temp = float(expInfo['pain_threshold']) + 1.5 # = 39.5 with a calibrated pain threshold of 38 °C

stimuli = StimuliFunction(
    duration, frequencies, amplitudes,
    random_phase=True, seed=seed)
stimuli.add_baseline_temp(baseline_temp)
stimuli.add_plateaus(n_plateaus=4, duration_per_plateau=15)

stimuli_length = stimuli.wave.shape[0] / stimuli.sample_rate
text_welcome = visual.TextStim(win=win, name='text_welcome',
    text='Willkommen zum Experiment!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
text_start_exp = visual.TextStim(win=win, name='text_start_exp',
    text='Das Experiment beginnt mit Tastendruck.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
key_welcome = keyboard.Keyboard()

# --- Initialize components for Routine "trial_vas_continuous" ---
slider_vas_cont = visual.Slider(win=win, name='slider_vas_cont',
    startValue=50, size=(1.0, 0.1), pos=(0, -0.3), units=win.units,
    labels=("Kein Schmerz","","","","","","","","","Stärkste vorstellbare Schmerzen"), ticks=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), granularity=0.0,
    style='rating', styleTweaks=('triangleMarker',), opacity=None,
    labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, ori=0.0, depth=0, readOnly=False)
polygon_cross_cont = visual.ShapeStim(
    win=win, name='polygon_cross_cont', vertices='cross',
    size=(0.1, 0.1),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
text_vas_cont = visual.TextStim(win=win, name='text_vas_cont',
    text='',
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# --- Initialize components for Routine "blank500" ---

# --- Initialize components for Routine "goodbye_screen" ---
text_goodbye = visual.TextStim(win=win, name='text_goodbye',
    text='Vielen Dank für Ihre Versuchsteilnahme!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "welcome_screen" ---
continueRoutine = True
# update component parameters for each repeat
key_welcome.keys = []
key_welcome.rt = []
_key_welcome_allKeys = []
# keep track of which components have finished
welcome_screenComponents = [text_welcome, text_start_exp, key_welcome]
for thisComponent in welcome_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "welcome_screen" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_welcome* updates
    
    # if text_welcome is starting this frame...
    if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_welcome.frameNStart = frameN  # exact frame index
        text_welcome.tStart = t  # local t and not account for scr refresh
        text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_welcome.started')
        # update status
        text_welcome.status = STARTED
        text_welcome.setAutoDraw(True)
    
    # if text_welcome is active this frame...
    if text_welcome.status == STARTED:
        # update params
        pass
    
    # if text_welcome is stopping this frame...
    if text_welcome.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_welcome.tStartRefresh + 1-frameTolerance:
            # keep track of stop time/frame for later
            text_welcome.tStop = t  # not accounting for scr refresh
            text_welcome.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome.stopped')
            # update status
            text_welcome.status = FINISHED
            text_welcome.setAutoDraw(False)
    
    # *text_start_exp* updates
    
    # if text_start_exp is starting this frame...
    if text_start_exp.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
        # keep track of start time/frame for later
        text_start_exp.frameNStart = frameN  # exact frame index
        text_start_exp.tStart = t  # local t and not account for scr refresh
        text_start_exp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_start_exp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_start_exp.started')
        # update status
        text_start_exp.status = STARTED
        text_start_exp.setAutoDraw(True)
    
    # if text_start_exp is active this frame...
    if text_start_exp.status == STARTED:
        # update params
        pass
    
    # *key_welcome* updates
    waitOnFlip = False
    
    # if key_welcome is starting this frame...
    if key_welcome.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        key_welcome.frameNStart = frameN  # exact frame index
        key_welcome.tStart = t  # local t and not account for scr refresh
        key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_welcome.started')
        # update status
        key_welcome.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_welcome.status == STARTED and not waitOnFlip:
        theseKeys = key_welcome.getKeys(keyList=['j','n','left','right','space'], waitRelease=False)
        _key_welcome_allKeys.extend(theseKeys)
        if len(_key_welcome_allKeys):
            key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
            key_welcome.rt = _key_welcome_allKeys[-1].rt
            key_welcome.duration = _key_welcome_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in welcome_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "welcome_screen" ---
for thisComponent in welcome_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_welcome.keys in ['', [], None]:  # No response was made
    key_welcome.keys = None
thisExp.addData('key_welcome.keys',key_welcome.keys)
if key_welcome.keys != None:  # we had a response
    thisExp.addData('key_welcome.rt', key_welcome.rt)
    thisExp.addData('key_welcome.duration', key_welcome.duration)
thisExp.nextEntry()
# the Routine "welcome_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "trial_vas_continuous" ---
continueRoutine = True
# update component parameters for each repeat
slider_vas_cont.reset()
# Run 'Begin Routine' code from mouse
"""Holds down the left mouse button"""
mouse.press(button="left")
# keep track of which components have finished
trial_vas_continuousComponents = [slider_vas_cont, polygon_cross_cont, text_vas_cont]
for thisComponent in trial_vas_continuousComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "trial_vas_continuous" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *slider_vas_cont* updates
    
    # if slider_vas_cont is starting this frame...
    if slider_vas_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        slider_vas_cont.frameNStart = frameN  # exact frame index
        slider_vas_cont.tStart = t  # local t and not account for scr refresh
        slider_vas_cont.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(slider_vas_cont, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'slider_vas_cont.started')
        # update status
        slider_vas_cont.status = STARTED
        slider_vas_cont.setAutoDraw(True)
    
    # if slider_vas_cont is active this frame...
    if slider_vas_cont.status == STARTED:
        # update params
        pass
    
    # if slider_vas_cont is stopping this frame...
    if slider_vas_cont.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > slider_vas_cont.tStartRefresh + stimuli_length-frameTolerance:
            # keep track of stop time/frame for later
            slider_vas_cont.tStop = t  # not accounting for scr refresh
            slider_vas_cont.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_vas_cont.stopped')
            # update status
            slider_vas_cont.status = FINISHED
            slider_vas_cont.setAutoDraw(False)
    # Run 'Each Frame' code from mouse
    """Check if mouse is pressed"""
    #mouse_limited.draw()
    
    if not mouse.is_pressed(button='left'):
        mouse.press(button='left')
    
    # TODO: Add movement contrains by resetting the y-coordinate of the mouse
    # to the slider. That way the mouse device can be moved freely and accidental
    # click won't cause any problems.
    # TODO: Find a way to represent pixel coordinates of the slider.
    
    # *polygon_cross_cont* updates
    
    # if polygon_cross_cont is starting this frame...
    if polygon_cross_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        polygon_cross_cont.frameNStart = frameN  # exact frame index
        polygon_cross_cont.tStart = t  # local t and not account for scr refresh
        polygon_cross_cont.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(polygon_cross_cont, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'polygon_cross_cont.started')
        # update status
        polygon_cross_cont.status = STARTED
        polygon_cross_cont.setAutoDraw(True)
    
    # if polygon_cross_cont is active this frame...
    if polygon_cross_cont.status == STARTED:
        # update params
        pass
    
    # if polygon_cross_cont is stopping this frame...
    if polygon_cross_cont.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > polygon_cross_cont.tStartRefresh + stimuli_length-frameTolerance:
            # keep track of stop time/frame for later
            polygon_cross_cont.tStop = t  # not accounting for scr refresh
            polygon_cross_cont.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon_cross_cont.stopped')
            # update status
            polygon_cross_cont.status = FINISHED
            polygon_cross_cont.setAutoDraw(False)
    
    # *text_vas_cont* updates
    
    # if text_vas_cont is starting this frame...
    if text_vas_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_vas_cont.frameNStart = frameN  # exact frame index
        text_vas_cont.tStart = t  # local t and not account for scr refresh
        text_vas_cont.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_vas_cont, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_vas_cont.started')
        # update status
        text_vas_cont.status = STARTED
        text_vas_cont.setAutoDraw(True)
    
    # if text_vas_cont is active this frame...
    if text_vas_cont.status == STARTED:
        # update params
        text_vas_cont.setText(f"Dies ist eine kontinuierliche VAS.\n\nIhr momentaner Schmerz liegt bei: {slider_vas_cont.getMarkerPos():.0f}" if slider_vas_cont.getMarkerPos() is not None else "Dies ist eine kontinuierliche VAS.\n\nIhr momentaner  Schmerz liegt bei: --", log=False)
    
    # if text_vas_cont is stopping this frame...
    if text_vas_cont.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_vas_cont.tStartRefresh + stimuli_length-frameTolerance:
            # keep track of stop time/frame for later
            text_vas_cont.tStop = t  # not accounting for scr refresh
            text_vas_cont.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_vas_cont.stopped')
            # update status
            text_vas_cont.status = FINISHED
            text_vas_cont.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trial_vas_continuousComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "trial_vas_continuous" ---
for thisComponent in trial_vas_continuousComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('slider_vas_cont.response', slider_vas_cont.getRating())
thisExp.addData('slider_vas_cont.rt', slider_vas_cont.getRT())
# Run 'End Routine' code from mouse
"""Releases the left mouse button"""
mouse.release(button="left")
# the Routine "trial_vas_continuous" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "blank500" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
blank500Components = []
for thisComponent in blank500Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "blank500" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in blank500Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "blank500" ---
for thisComponent in blank500Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "blank500" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "goodbye_screen" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
goodbye_screenComponents = [text_goodbye]
for thisComponent in goodbye_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "goodbye_screen" ---
routineForceEnded = not continueRoutine
while continueRoutine and routineTimer.getTime() < 1.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_goodbye* updates
    
    # if text_goodbye is starting this frame...
    if text_goodbye.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        text_goodbye.frameNStart = frameN  # exact frame index
        text_goodbye.tStart = t  # local t and not account for scr refresh
        text_goodbye.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_goodbye, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_goodbye.started')
        # update status
        text_goodbye.status = STARTED
        text_goodbye.setAutoDraw(True)
    
    # if text_goodbye is active this frame...
    if text_goodbye.status == STARTED:
        # update params
        pass
    
    # if text_goodbye is stopping this frame...
    if text_goodbye.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_goodbye.tStartRefresh + 1-frameTolerance:
            # keep track of stop time/frame for later
            text_goodbye.tStop = t  # not accounting for scr refresh
            text_goodbye.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_goodbye.stopped')
            # update status
            text_goodbye.status = FINISHED
            text_goodbye.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in goodbye_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "goodbye_screen" ---
for thisComponent in goodbye_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-1.000000)

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
