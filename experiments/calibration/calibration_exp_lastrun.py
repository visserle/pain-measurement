#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.1),
    on Tue Nov 14 17:06:39 2023
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
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.1'
expName = 'calibration_exp'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'age': '20',
    'gender': 'Female',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Run 'Before Experiment' code from all_variables
from src.experiments.thermoino_dummy import Thermoino

# Logger
from pathlib import Path
from datetime import datetime
from src.experiments.log_config import configure_logging, close_root_logging

# Configure logging
log_dir = Path('log')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
log_file = log_dir / log_filename_str

configure_logging(log_file=log_file)

# Thermoino
port = "COM3"
# COM7 for top usb port on the front, use list_com_ports() to find out
mms_baseline = 30 # has to be the same as in MMS
mms_rate_of_rise = 10 # has to be the same as in MMS

# Stimuli
stimuli_clock = core.Clock()
stimuli_duration = 8
iti_duration = 8  + np.random.randint(2, 4)
iti_duration_short = 2
cross_size = (0.06, 0.06)

# Pre-exposure
temps_preexposure = [35, 36, 37]
correction_after_preexposure = 2 # we substract from temp_start_vas70

# Estimator
trials_vas70 = 7 # is the same as nReps in psychopy loop
temp_start_vas70 = 42
temp_std_vas70 = 3.5

trials_vas0 = 5 # is the same as nReps in psychopy loop
temp_start_vas0 = None #  will be set after VAS 0 estimate
temp_start_vas0_minus = 3 # VAS 0 estimate plus int
temp_std_vas0 = 1.5 # smaller std for higher temperatures

# Run 'Before Experiment' code from estimator_vas70
from src.experiments.calibration import BayesianEstimatorVAS

# Estimator for VAS 0 (pain threshold)
estimator_vas70 = BayesianEstimatorVAS(
    vas_value= 70, 
    temp_start=temp_start_vas70,
    temp_std=temp_std_vas70,
    trials=trials_vas70)

# Run 'Before Experiment' code from estimator_vas0
# instantiating here does not make sense because we need the values from the VAS 0 esitmator first
# Run 'Before Experiment' code from save_participant_data
from src.experiments.participant_data import add_participant

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/visser/drive/PhD/Code/mpad-pilot/experiments/calibration/calibration_exp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file
    # return log file
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[600, 600], fullscr=False, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
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
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    # Run 'Begin Experiment' code from thermoino
    luigi = Thermoino(
        port=port,
        mms_baseline=mms_baseline, # has to be the same as in MMS
        mms_rate_of_rise=mms_rate_of_rise) # has to be the same as in MMS
    
    luigi.connect()
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Herzlich willkommen zum Experiment!\n\n\nBitte drücken Sie die Leertaste.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_2" ---
    text_welcome_2 = visual.TextStim(win=win, name='text_welcome_2',
        text='Wir beginnen das Experiment mit einer Schmerz-Kalibrierung.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_3" ---
    text_welcome_3 = visual.TextStim(win=win, name='text_welcome_3',
        text='Hierzu wärmen wir die Hautstellte an Ihrem Arm zuerst kurz auf.\n\nAnschließend bestimmen wir, wann Sie leichte und starke Schmerzen verspüren.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "info_preexposure" ---
    text_info_preexposure = visual.TextStim(win=win, name='text_info_preexposure',
        text='Wir beginnen mit dem Aufwärmen der Hautstelle.\nHierbei müssen Sie nichts weiter tun.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_info_preexposure = keyboard.Keyboard()
    
    # --- Initialize components for Routine "iti_short" ---
    cross_neutral_short = visual.ShapeStim(
        win=win, name='cross_neutral_short', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "preexposure" ---
    corss_pain_preexposure = visual.ShapeStim(
        win=win, name='corss_pain_preexposure', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_preexposure" ---
    question_preexposure = visual.TextStim(win=win, name='question_preexposure',
        text='War einer dieser Reize für Sie schmerzhaft?\n\n(Drücken Sie "y" für Ja oder "n" für Nein.)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    response_preexposure = keyboard.Keyboard()
    
    # --- Initialize components for Routine "count_in_preexposure" ---
    text_answer_preexposure = visual.TextStim(win=win, name='text_answer_preexposure',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "info_vas70" ---
    text_info_vas70 = visual.TextStim(win=win, name='text_info_vas70',
        text='Als Nächstes möchten wir herausfinden, ab wann Sie starke Schmerzen verspüren.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_info_vas70 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "info_vas70_2" ---
    text_info_vas70_2 = visual.TextStim(win=win, name='text_info_vas70_2',
        text='Dabei orientieren wir uns an einer Schmerz-Skala von 1 bis 10:\n\n\n\n\n\n\n\n\nUnser Ziel ist es, herauszufinden, ab wann Sie eine 7 von 10 (starke / sehr starke Schmerzen) verspüren.\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_info_vas70_2 = keyboard.Keyboard()
    img_vas = visual.ImageStim(
        win=win,
        name='img_vas', 
        image='vas_7.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.1), size=(1, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "info_vas_70_3" ---
    text_info_vas70_3 = visual.TextStim(win=win, name='text_info_vas70_3',
        text='Dank dem Capsaicin ist Ihre Schmerzschwelle dabei nach unten verlagert.\n\nDas bedeutet, Sie verspüren schneller starken Schmerz.\nAber keine Sorge: Zu keinem Zeitpunkt ist Ihre Haut durch Verbrennungen oder Ähnliches bedroht.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_vas_70_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_vas70" ---
    cross_pain_vas70 = visual.ShapeStim(
        win=win, name='cross_pain_vas70', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_vas70" ---
    question_vas70 = visual.TextStim(win=win, name='question_vas70',
        text='War dieser Reiz mindestens eine 7 von 10 (starker / sehr starker Schmerz)?\n\n(y/n)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    response_vas70 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "estimate_vas70" ---
    answer_vas70 = visual.TextStim(win=win, name='answer_vas70',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "info_vas0_transition" ---
    text_info_vas0 = visual.TextStim(win=win, name='text_info_vas0',
        text='Wunderbar!\n\n\nAls Nächstes möchten wir bestimmen, wo Ihre Schmerzwelle liegt - also ab wann Sie erste Schmerzen verspüren.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_info_vas0 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_vas0" ---
    cross_pain_vas0 = visual.ShapeStim(
        win=win, name='cross_pain_vas0', vertices='cross',
        size=cross_size,
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_vas0" ---
    question_vas0 = visual.TextStim(win=win, name='question_vas0',
        text='War dieser Reiz für Sie schmerzhaft?\n\n(y/n)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    response_vas0 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "estimate_vas0" ---
    answer_vas0 = visual.TextStim(win=win, name='answer_vas0',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "bye" ---
    text_bye = visual.TextStim(win=win, name='text_bye',
        text='Vielen Dank!\n\nAls Nächstes geht es mit dem Hauptexperiment weiter.\nMelden Sie sich bitte bei der Versuchsleitung.\n\n\n(Leertaste drücken, um Kalibrierung zu beenden)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_bye = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [text_welcome, key_welcome]
    for thisComponent in welcomeComponents:
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
    
    # --- Run Routine "welcome" ---
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
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
            theseKeys = key_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome_2.started', globalClock.getTime())
    key_welcome_2.keys = []
    key_welcome_2.rt = []
    _key_welcome_2_allKeys = []
    # keep track of which components have finished
    welcome_2Components = [text_welcome_2, key_welcome_2]
    for thisComponent in welcome_2Components:
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
    
    # --- Run Routine "welcome_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome_2* updates
        
        # if text_welcome_2 is starting this frame...
        if text_welcome_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome_2.frameNStart = frameN  # exact frame index
            text_welcome_2.tStart = t  # local t and not account for scr refresh
            text_welcome_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome_2.started')
            # update status
            text_welcome_2.status = STARTED
            text_welcome_2.setAutoDraw(True)
        
        # if text_welcome_2 is active this frame...
        if text_welcome_2.status == STARTED:
            # update params
            pass
        
        # *key_welcome_2* updates
        waitOnFlip = False
        
        # if key_welcome_2 is starting this frame...
        if key_welcome_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_welcome_2.frameNStart = frameN  # exact frame index
            key_welcome_2.tStart = t  # local t and not account for scr refresh
            key_welcome_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_welcome_2.started')
            # update status
            key_welcome_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome_2.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_2_allKeys.extend(theseKeys)
            if len(_key_welcome_2_allKeys):
                key_welcome_2.keys = _key_welcome_2_allKeys[-1].name  # just the last key pressed
                key_welcome_2.rt = _key_welcome_2_allKeys[-1].rt
                key_welcome_2.duration = _key_welcome_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome_2" ---
    for thisComponent in welcome_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome_2.stopped', globalClock.getTime())
    # check responses
    if key_welcome_2.keys in ['', [], None]:  # No response was made
        key_welcome_2.keys = None
    thisExp.addData('key_welcome_2.keys',key_welcome_2.keys)
    if key_welcome_2.keys != None:  # we had a response
        thisExp.addData('key_welcome_2.rt', key_welcome_2.rt)
        thisExp.addData('key_welcome_2.duration', key_welcome_2.duration)
    thisExp.nextEntry()
    # the Routine "welcome_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome_3.started', globalClock.getTime())
    key_welcome_3.keys = []
    key_welcome_3.rt = []
    _key_welcome_3_allKeys = []
    # keep track of which components have finished
    welcome_3Components = [text_welcome_3, key_welcome_3]
    for thisComponent in welcome_3Components:
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
    
    # --- Run Routine "welcome_3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome_3* updates
        
        # if text_welcome_3 is starting this frame...
        if text_welcome_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome_3.frameNStart = frameN  # exact frame index
            text_welcome_3.tStart = t  # local t and not account for scr refresh
            text_welcome_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome_3.started')
            # update status
            text_welcome_3.status = STARTED
            text_welcome_3.setAutoDraw(True)
        
        # if text_welcome_3 is active this frame...
        if text_welcome_3.status == STARTED:
            # update params
            pass
        
        # *key_welcome_3* updates
        waitOnFlip = False
        
        # if key_welcome_3 is starting this frame...
        if key_welcome_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_welcome_3.frameNStart = frameN  # exact frame index
            key_welcome_3.tStart = t  # local t and not account for scr refresh
            key_welcome_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_welcome_3.started')
            # update status
            key_welcome_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome_3.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome_3.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_3_allKeys.extend(theseKeys)
            if len(_key_welcome_3_allKeys):
                key_welcome_3.keys = _key_welcome_3_allKeys[-1].name  # just the last key pressed
                key_welcome_3.rt = _key_welcome_3_allKeys[-1].rt
                key_welcome_3.duration = _key_welcome_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome_3" ---
    for thisComponent in welcome_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome_3.stopped', globalClock.getTime())
    # check responses
    if key_welcome_3.keys in ['', [], None]:  # No response was made
        key_welcome_3.keys = None
    thisExp.addData('key_welcome_3.keys',key_welcome_3.keys)
    if key_welcome_3.keys != None:  # we had a response
        thisExp.addData('key_welcome_3.rt', key_welcome_3.rt)
        thisExp.addData('key_welcome_3.duration', key_welcome_3.duration)
    thisExp.nextEntry()
    # the Routine "welcome_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "info_preexposure" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('info_preexposure.started', globalClock.getTime())
    key_info_preexposure.keys = []
    key_info_preexposure.rt = []
    _key_info_preexposure_allKeys = []
    # keep track of which components have finished
    info_preexposureComponents = [text_info_preexposure, key_info_preexposure]
    for thisComponent in info_preexposureComponents:
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
    
    # --- Run Routine "info_preexposure" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_info_preexposure* updates
        
        # if text_info_preexposure is starting this frame...
        if text_info_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_info_preexposure.frameNStart = frameN  # exact frame index
            text_info_preexposure.tStart = t  # local t and not account for scr refresh
            text_info_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_info_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_info_preexposure.started')
            # update status
            text_info_preexposure.status = STARTED
            text_info_preexposure.setAutoDraw(True)
        
        # if text_info_preexposure is active this frame...
        if text_info_preexposure.status == STARTED:
            # update params
            pass
        
        # *key_info_preexposure* updates
        waitOnFlip = False
        
        # if key_info_preexposure is starting this frame...
        if key_info_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_info_preexposure.frameNStart = frameN  # exact frame index
            key_info_preexposure.tStart = t  # local t and not account for scr refresh
            key_info_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_info_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_info_preexposure.started')
            # update status
            key_info_preexposure.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_info_preexposure.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_info_preexposure.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_info_preexposure.status == STARTED and not waitOnFlip:
            theseKeys = key_info_preexposure.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_info_preexposure_allKeys.extend(theseKeys)
            if len(_key_info_preexposure_allKeys):
                key_info_preexposure.keys = _key_info_preexposure_allKeys[-1].name  # just the last key pressed
                key_info_preexposure.rt = _key_info_preexposure_allKeys[-1].rt
                key_info_preexposure.duration = _key_info_preexposure_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in info_preexposureComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "info_preexposure" ---
    for thisComponent in info_preexposureComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('info_preexposure.stopped', globalClock.getTime())
    # check responses
    if key_info_preexposure.keys in ['', [], None]:  # No response was made
        key_info_preexposure.keys = None
    thisExp.addData('key_info_preexposure.keys',key_info_preexposure.keys)
    if key_info_preexposure.keys != None:  # we had a response
        thisExp.addData('key_info_preexposure.rt', key_info_preexposure.rt)
        thisExp.addData('key_info_preexposure.duration', key_info_preexposure.duration)
    thisExp.nextEntry()
    # the Routine "info_preexposure" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "iti_short" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('iti_short.started', globalClock.getTime())
    # keep track of which components have finished
    iti_shortComponents = [cross_neutral_short]
    for thisComponent in iti_shortComponents:
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
    
    # --- Run Routine "iti_short" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross_neutral_short* updates
        
        # if cross_neutral_short is starting this frame...
        if cross_neutral_short.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            cross_neutral_short.frameNStart = frameN  # exact frame index
            cross_neutral_short.tStart = t  # local t and not account for scr refresh
            cross_neutral_short.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross_neutral_short, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross_neutral_short.started')
            # update status
            cross_neutral_short.status = STARTED
            cross_neutral_short.setAutoDraw(True)
        
        # if cross_neutral_short is active this frame...
        if cross_neutral_short.status == STARTED:
            # update params
            pass
        
        # if cross_neutral_short is stopping this frame...
        if cross_neutral_short.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross_neutral_short.tStartRefresh + iti_duration_short-frameTolerance:
                # keep track of stop time/frame for later
                cross_neutral_short.tStop = t  # not accounting for scr refresh
                cross_neutral_short.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_neutral_short.stopped')
                # update status
                cross_neutral_short.status = FINISHED
                cross_neutral_short.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in iti_shortComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "iti_short" ---
    for thisComponent in iti_shortComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('iti_short.stopped', globalClock.getTime())
    # the Routine "iti_short" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_preexposure = data.TrialHandler(nReps=len(temps_preexposure), method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_preexposure')
    thisExp.addLoop(loop_preexposure)  # add the loop to the experiment
    thisLoop_preexposure = loop_preexposure.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_preexposure.rgb)
    if thisLoop_preexposure != None:
        for paramName in thisLoop_preexposure:
            globals()[paramName] = thisLoop_preexposure[paramName]
    
    for thisLoop_preexposure in loop_preexposure:
        currentLoop = loop_preexposure
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_preexposure.rgb)
        if thisLoop_preexposure != None:
            for paramName in thisLoop_preexposure:
                globals()[paramName] = thisLoop_preexposure[paramName]
        
        # --- Prepare to start Routine "iti" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('iti.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (loop_preexposure.thisN == 0)
        # keep track of which components have finished
        itiComponents = [cross_neutral]
        for thisComponent in itiComponents:
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
        
        # --- Run Routine "iti" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_neutral* updates
            
            # if cross_neutral is starting this frame...
            if cross_neutral.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cross_neutral.frameNStart = frameN  # exact frame index
                cross_neutral.tStart = t  # local t and not account for scr refresh
                cross_neutral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_neutral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_neutral.started')
                # update status
                cross_neutral.status = STARTED
                cross_neutral.setAutoDraw(True)
            
            # if cross_neutral is active this frame...
            if cross_neutral.status == STARTED:
                # update params
                pass
            
            # if cross_neutral is stopping this frame...
            if cross_neutral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_neutral.tStartRefresh + iti_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_neutral.tStop = t  # not accounting for scr refresh
                    cross_neutral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_neutral.stopped')
                    # update status
                    cross_neutral.status = FINISHED
                    cross_neutral.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in itiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in itiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('iti.stopped', globalClock.getTime())
        # the Routine "iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preexposure" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preexposure.started', globalClock.getTime())
        # Run 'Begin Routine' code from thermoino_prexposure
        checked = False
        stimuli_clock.reset()
        
        trial = loop_preexposure.thisN
        luigi.trigger()
        time_for_ramp_up = luigi.set_temp(temps_preexposure[trial])[1]
        # keep track of which components have finished
        preexposureComponents = [corss_pain_preexposure]
        for thisComponent in preexposureComponents:
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
        
        # --- Run Routine "preexposure" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from thermoino_prexposure
            if not checked:
                if stimuli_clock.getTime() > (time_for_ramp_up + stimuli_duration):
                    luigi.set_temp(mms_baseline)
                    checked = True
            
            # *corss_pain_preexposure* updates
            
            # if corss_pain_preexposure is starting this frame...
            if corss_pain_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                corss_pain_preexposure.frameNStart = frameN  # exact frame index
                corss_pain_preexposure.tStart = t  # local t and not account for scr refresh
                corss_pain_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(corss_pain_preexposure, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'corss_pain_preexposure.started')
                # update status
                corss_pain_preexposure.status = STARTED
                corss_pain_preexposure.setAutoDraw(True)
            
            # if corss_pain_preexposure is active this frame...
            if corss_pain_preexposure.status == STARTED:
                # update params
                pass
            
            # if corss_pain_preexposure is stopping this frame...
            if corss_pain_preexposure.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > corss_pain_preexposure.tStartRefresh + stimuli_duration + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    corss_pain_preexposure.tStop = t  # not accounting for scr refresh
                    corss_pain_preexposure.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'corss_pain_preexposure.stopped')
                    # update status
                    corss_pain_preexposure.status = FINISHED
                    corss_pain_preexposure.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preexposureComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preexposure" ---
        for thisComponent in preexposureComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preexposure.stopped', globalClock.getTime())
        # the Routine "preexposure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed len(temps_preexposure) repeats of 'loop_preexposure'
    
    
    # --- Prepare to start Routine "feedback_preexposure" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('feedback_preexposure.started', globalClock.getTime())
    response_preexposure.keys = []
    response_preexposure.rt = []
    _response_preexposure_allKeys = []
    # keep track of which components have finished
    feedback_preexposureComponents = [question_preexposure, response_preexposure]
    for thisComponent in feedback_preexposureComponents:
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
    
    # --- Run Routine "feedback_preexposure" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_preexposure* updates
        
        # if question_preexposure is starting this frame...
        if question_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_preexposure.frameNStart = frameN  # exact frame index
            question_preexposure.tStart = t  # local t and not account for scr refresh
            question_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'question_preexposure.started')
            # update status
            question_preexposure.status = STARTED
            question_preexposure.setAutoDraw(True)
        
        # if question_preexposure is active this frame...
        if question_preexposure.status == STARTED:
            # update params
            pass
        
        # *response_preexposure* updates
        waitOnFlip = False
        
        # if response_preexposure is starting this frame...
        if response_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            response_preexposure.frameNStart = frameN  # exact frame index
            response_preexposure.tStart = t  # local t and not account for scr refresh
            response_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(response_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'response_preexposure.started')
            # update status
            response_preexposure.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(response_preexposure.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(response_preexposure.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if response_preexposure.status == STARTED and not waitOnFlip:
            theseKeys = response_preexposure.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
            _response_preexposure_allKeys.extend(theseKeys)
            if len(_response_preexposure_allKeys):
                response_preexposure.keys = _response_preexposure_allKeys[-1].name  # just the last key pressed
                response_preexposure.rt = _response_preexposure_allKeys[-1].rt
                response_preexposure.duration = _response_preexposure_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedback_preexposureComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "feedback_preexposure" ---
    for thisComponent in feedback_preexposureComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('feedback_preexposure.stopped', globalClock.getTime())
    # check responses
    if response_preexposure.keys in ['', [], None]:  # No response was made
        response_preexposure.keys = None
    thisExp.addData('response_preexposure.keys',response_preexposure.keys)
    if response_preexposure.keys != None:  # we had a response
        thisExp.addData('response_preexposure.rt', response_preexposure.rt)
        thisExp.addData('response_preexposure.duration', response_preexposure.duration)
    thisExp.nextEntry()
    # the Routine "feedback_preexposure" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "count_in_preexposure" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('count_in_preexposure.started', globalClock.getTime())
    text_answer_preexposure.setText("Ihre Antwort war: Nein." if response_preexposure.keys == "n" else "Ihre Antwort war: Ja."
    )
    # keep track of which components have finished
    count_in_preexposureComponents = [text_answer_preexposure]
    for thisComponent in count_in_preexposureComponents:
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
    
    # --- Run Routine "count_in_preexposure" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_answer_preexposure* updates
        
        # if text_answer_preexposure is starting this frame...
        if text_answer_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_answer_preexposure.frameNStart = frameN  # exact frame index
            text_answer_preexposure.tStart = t  # local t and not account for scr refresh
            text_answer_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_answer_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_answer_preexposure.started')
            # update status
            text_answer_preexposure.status = STARTED
            text_answer_preexposure.setAutoDraw(True)
        
        # if text_answer_preexposure is active this frame...
        if text_answer_preexposure.status == STARTED:
            # update params
            pass
        
        # if text_answer_preexposure is stopping this frame...
        if text_answer_preexposure.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_answer_preexposure.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_answer_preexposure.tStop = t  # not accounting for scr refresh
                text_answer_preexposure.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_preexposure.stopped')
                # update status
                text_answer_preexposure.status = FINISHED
                text_answer_preexposure.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in count_in_preexposureComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "count_in_preexposure" ---
    for thisComponent in count_in_preexposureComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('count_in_preexposure.stopped', globalClock.getTime())
    # Run 'End Routine' code from transition_preexposure_to_vas70
    # If response was yes
    logging.info("Preexposure painful? Answer: %s", response_preexposure.keys)
    
    if response_preexposure.keys == "y":
        # Decrease starting temperature
        global temp_start_vas70 # psychopy has to be able to find it in the spaghetti
        temp_start_vas70 -= correction_after_preexposure
        # Reinitialize estimator for VAS 70 with different temp_start
        global estimator_vas70
        estimator_vas70 = BayesianEstimatorVAS(
            vas_value=70,
            temp_start=temp_start_vas70,
            temp_std=temp_std_vas70,
            trials=trials_vas70)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # --- Prepare to start Routine "info_vas70" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('info_vas70.started', globalClock.getTime())
    key_info_vas70.keys = []
    key_info_vas70.rt = []
    _key_info_vas70_allKeys = []
    # keep track of which components have finished
    info_vas70Components = [text_info_vas70, key_info_vas70]
    for thisComponent in info_vas70Components:
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
    
    # --- Run Routine "info_vas70" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_info_vas70* updates
        
        # if text_info_vas70 is starting this frame...
        if text_info_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_info_vas70.frameNStart = frameN  # exact frame index
            text_info_vas70.tStart = t  # local t and not account for scr refresh
            text_info_vas70.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_info_vas70, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_info_vas70.started')
            # update status
            text_info_vas70.status = STARTED
            text_info_vas70.setAutoDraw(True)
        
        # if text_info_vas70 is active this frame...
        if text_info_vas70.status == STARTED:
            # update params
            pass
        
        # *key_info_vas70* updates
        waitOnFlip = False
        
        # if key_info_vas70 is starting this frame...
        if key_info_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_info_vas70.frameNStart = frameN  # exact frame index
            key_info_vas70.tStart = t  # local t and not account for scr refresh
            key_info_vas70.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_info_vas70, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_info_vas70.started')
            # update status
            key_info_vas70.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_info_vas70.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_info_vas70.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_info_vas70.status == STARTED and not waitOnFlip:
            theseKeys = key_info_vas70.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_info_vas70_allKeys.extend(theseKeys)
            if len(_key_info_vas70_allKeys):
                key_info_vas70.keys = _key_info_vas70_allKeys[-1].name  # just the last key pressed
                key_info_vas70.rt = _key_info_vas70_allKeys[-1].rt
                key_info_vas70.duration = _key_info_vas70_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in info_vas70Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "info_vas70" ---
    for thisComponent in info_vas70Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('info_vas70.stopped', globalClock.getTime())
    # check responses
    if key_info_vas70.keys in ['', [], None]:  # No response was made
        key_info_vas70.keys = None
    thisExp.addData('key_info_vas70.keys',key_info_vas70.keys)
    if key_info_vas70.keys != None:  # we had a response
        thisExp.addData('key_info_vas70.rt', key_info_vas70.rt)
        thisExp.addData('key_info_vas70.duration', key_info_vas70.duration)
    thisExp.nextEntry()
    # the Routine "info_vas70" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "info_vas70_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('info_vas70_2.started', globalClock.getTime())
    key_info_vas70_2.keys = []
    key_info_vas70_2.rt = []
    _key_info_vas70_2_allKeys = []
    # keep track of which components have finished
    info_vas70_2Components = [text_info_vas70_2, key_info_vas70_2, img_vas]
    for thisComponent in info_vas70_2Components:
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
    
    # --- Run Routine "info_vas70_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_info_vas70_2* updates
        
        # if text_info_vas70_2 is starting this frame...
        if text_info_vas70_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_info_vas70_2.frameNStart = frameN  # exact frame index
            text_info_vas70_2.tStart = t  # local t and not account for scr refresh
            text_info_vas70_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_info_vas70_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_info_vas70_2.started')
            # update status
            text_info_vas70_2.status = STARTED
            text_info_vas70_2.setAutoDraw(True)
        
        # if text_info_vas70_2 is active this frame...
        if text_info_vas70_2.status == STARTED:
            # update params
            pass
        
        # *key_info_vas70_2* updates
        waitOnFlip = False
        
        # if key_info_vas70_2 is starting this frame...
        if key_info_vas70_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_info_vas70_2.frameNStart = frameN  # exact frame index
            key_info_vas70_2.tStart = t  # local t and not account for scr refresh
            key_info_vas70_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_info_vas70_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_info_vas70_2.started')
            # update status
            key_info_vas70_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_info_vas70_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_info_vas70_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_info_vas70_2.status == STARTED and not waitOnFlip:
            theseKeys = key_info_vas70_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_info_vas70_2_allKeys.extend(theseKeys)
            if len(_key_info_vas70_2_allKeys):
                key_info_vas70_2.keys = _key_info_vas70_2_allKeys[-1].name  # just the last key pressed
                key_info_vas70_2.rt = _key_info_vas70_2_allKeys[-1].rt
                key_info_vas70_2.duration = _key_info_vas70_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *img_vas* updates
        
        # if img_vas is starting this frame...
        if img_vas.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            img_vas.frameNStart = frameN  # exact frame index
            img_vas.tStart = t  # local t and not account for scr refresh
            img_vas.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(img_vas, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'img_vas.started')
            # update status
            img_vas.status = STARTED
            img_vas.setAutoDraw(True)
        
        # if img_vas is active this frame...
        if img_vas.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in info_vas70_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "info_vas70_2" ---
    for thisComponent in info_vas70_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('info_vas70_2.stopped', globalClock.getTime())
    # check responses
    if key_info_vas70_2.keys in ['', [], None]:  # No response was made
        key_info_vas70_2.keys = None
    thisExp.addData('key_info_vas70_2.keys',key_info_vas70_2.keys)
    if key_info_vas70_2.keys != None:  # we had a response
        thisExp.addData('key_info_vas70_2.rt', key_info_vas70_2.rt)
        thisExp.addData('key_info_vas70_2.duration', key_info_vas70_2.duration)
    thisExp.nextEntry()
    # the Routine "info_vas70_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "info_vas_70_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('info_vas_70_3.started', globalClock.getTime())
    key_vas_70_3.keys = []
    key_vas_70_3.rt = []
    _key_vas_70_3_allKeys = []
    # keep track of which components have finished
    info_vas_70_3Components = [text_info_vas70_3, key_vas_70_3]
    for thisComponent in info_vas_70_3Components:
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
    
    # --- Run Routine "info_vas_70_3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_info_vas70_3* updates
        
        # if text_info_vas70_3 is starting this frame...
        if text_info_vas70_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_info_vas70_3.frameNStart = frameN  # exact frame index
            text_info_vas70_3.tStart = t  # local t and not account for scr refresh
            text_info_vas70_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_info_vas70_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_info_vas70_3.started')
            # update status
            text_info_vas70_3.status = STARTED
            text_info_vas70_3.setAutoDraw(True)
        
        # if text_info_vas70_3 is active this frame...
        if text_info_vas70_3.status == STARTED:
            # update params
            pass
        
        # *key_vas_70_3* updates
        waitOnFlip = False
        
        # if key_vas_70_3 is starting this frame...
        if key_vas_70_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_vas_70_3.frameNStart = frameN  # exact frame index
            key_vas_70_3.tStart = t  # local t and not account for scr refresh
            key_vas_70_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_vas_70_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_vas_70_3.started')
            # update status
            key_vas_70_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_vas_70_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_vas_70_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_vas_70_3.status == STARTED and not waitOnFlip:
            theseKeys = key_vas_70_3.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_vas_70_3_allKeys.extend(theseKeys)
            if len(_key_vas_70_3_allKeys):
                key_vas_70_3.keys = _key_vas_70_3_allKeys[-1].name  # just the last key pressed
                key_vas_70_3.rt = _key_vas_70_3_allKeys[-1].rt
                key_vas_70_3.duration = _key_vas_70_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in info_vas_70_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "info_vas_70_3" ---
    for thisComponent in info_vas_70_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('info_vas_70_3.stopped', globalClock.getTime())
    # check responses
    if key_vas_70_3.keys in ['', [], None]:  # No response was made
        key_vas_70_3.keys = None
    thisExp.addData('key_vas_70_3.keys',key_vas_70_3.keys)
    if key_vas_70_3.keys != None:  # we had a response
        thisExp.addData('key_vas_70_3.rt', key_vas_70_3.rt)
        thisExp.addData('key_vas_70_3.duration', key_vas_70_3.duration)
    thisExp.nextEntry()
    # the Routine "info_vas_70_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_vas70 = data.TrialHandler(nReps=trials_vas70, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_vas70')
    thisExp.addLoop(loop_vas70)  # add the loop to the experiment
    thisLoop_vas70 = loop_vas70.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_vas70.rgb)
    if thisLoop_vas70 != None:
        for paramName in thisLoop_vas70:
            globals()[paramName] = thisLoop_vas70[paramName]
    
    for thisLoop_vas70 in loop_vas70:
        currentLoop = loop_vas70
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_vas70.rgb)
        if thisLoop_vas70 != None:
            for paramName in thisLoop_vas70:
                globals()[paramName] = thisLoop_vas70[paramName]
        
        # --- Prepare to start Routine "iti" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('iti.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (loop_preexposure.thisN == 0)
        # keep track of which components have finished
        itiComponents = [cross_neutral]
        for thisComponent in itiComponents:
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
        
        # --- Run Routine "iti" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_neutral* updates
            
            # if cross_neutral is starting this frame...
            if cross_neutral.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cross_neutral.frameNStart = frameN  # exact frame index
                cross_neutral.tStart = t  # local t and not account for scr refresh
                cross_neutral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_neutral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_neutral.started')
                # update status
                cross_neutral.status = STARTED
                cross_neutral.setAutoDraw(True)
            
            # if cross_neutral is active this frame...
            if cross_neutral.status == STARTED:
                # update params
                pass
            
            # if cross_neutral is stopping this frame...
            if cross_neutral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_neutral.tStartRefresh + iti_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_neutral.tStop = t  # not accounting for scr refresh
                    cross_neutral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_neutral.stopped')
                    # update status
                    cross_neutral.status = FINISHED
                    cross_neutral.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in itiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in itiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('iti.stopped', globalClock.getTime())
        # the Routine "iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_vas70" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_vas70.started', globalClock.getTime())
        # Run 'Begin Routine' code from thermoino_vas70
        checked = False
        stimuli_clock.reset()
        
        luigi.trigger()
        time_for_ramp_up = luigi.set_temp(estimator_vas70.current_temp)[1]
        # keep track of which components have finished
        trial_vas70Components = [cross_pain_vas70]
        for thisComponent in trial_vas70Components:
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
        
        # --- Run Routine "trial_vas70" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from thermoino_vas70
            if not checked:
                if stimuli_clock.getTime() > (time_for_ramp_up + stimuli_duration):
                    luigi.set_temp(mms_baseline)
                    checked = True
            
            # *cross_pain_vas70* updates
            
            # if cross_pain_vas70 is starting this frame...
            if cross_pain_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_pain_vas70.frameNStart = frameN  # exact frame index
                cross_pain_vas70.tStart = t  # local t and not account for scr refresh
                cross_pain_vas70.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_pain_vas70, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_pain_vas70.started')
                # update status
                cross_pain_vas70.status = STARTED
                cross_pain_vas70.setAutoDraw(True)
            
            # if cross_pain_vas70 is active this frame...
            if cross_pain_vas70.status == STARTED:
                # update params
                pass
            
            # if cross_pain_vas70 is stopping this frame...
            if cross_pain_vas70.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_pain_vas70.tStartRefresh + stimuli_duration + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_pain_vas70.tStop = t  # not accounting for scr refresh
                    cross_pain_vas70.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_pain_vas70.stopped')
                    # update status
                    cross_pain_vas70.status = FINISHED
                    cross_pain_vas70.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_vas70Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_vas70" ---
        for thisComponent in trial_vas70Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_vas70.stopped', globalClock.getTime())
        # the Routine "trial_vas70" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_vas70" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback_vas70.started', globalClock.getTime())
        response_vas70.keys = []
        response_vas70.rt = []
        _response_vas70_allKeys = []
        # keep track of which components have finished
        feedback_vas70Components = [question_vas70, response_vas70]
        for thisComponent in feedback_vas70Components:
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
        
        # --- Run Routine "feedback_vas70" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_vas70* updates
            
            # if question_vas70 is starting this frame...
            if question_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_vas70.frameNStart = frameN  # exact frame index
                question_vas70.tStart = t  # local t and not account for scr refresh
                question_vas70.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_vas70, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_vas70.started')
                # update status
                question_vas70.status = STARTED
                question_vas70.setAutoDraw(True)
            
            # if question_vas70 is active this frame...
            if question_vas70.status == STARTED:
                # update params
                pass
            
            # if question_vas70 is stopping this frame...
            if question_vas70.status == STARTED:
                if bool(response_vas70.status==FINISHED):
                    # keep track of stop time/frame for later
                    question_vas70.tStop = t  # not accounting for scr refresh
                    question_vas70.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_vas70.stopped')
                    # update status
                    question_vas70.status = FINISHED
                    question_vas70.setAutoDraw(False)
            
            # *response_vas70* updates
            waitOnFlip = False
            
            # if response_vas70 is starting this frame...
            if response_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                response_vas70.frameNStart = frameN  # exact frame index
                response_vas70.tStart = t  # local t and not account for scr refresh
                response_vas70.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(response_vas70, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'response_vas70.started')
                # update status
                response_vas70.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(response_vas70.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(response_vas70.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if response_vas70.status == STARTED and not waitOnFlip:
                theseKeys = response_vas70.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=True)
                _response_vas70_allKeys.extend(theseKeys)
                if len(_response_vas70_allKeys):
                    response_vas70.keys = _response_vas70_allKeys[-1].name  # just the last key pressed
                    response_vas70.rt = _response_vas70_allKeys[-1].rt
                    response_vas70.duration = _response_vas70_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_vas70Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_vas70" ---
        for thisComponent in feedback_vas70Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback_vas70.stopped', globalClock.getTime())
        # check responses
        if response_vas70.keys in ['', [], None]:  # No response was made
            response_vas70.keys = None
        loop_vas70.addData('response_vas70.keys',response_vas70.keys)
        if response_vas70.keys != None:  # we had a response
            loop_vas70.addData('response_vas70.rt', response_vas70.rt)
            loop_vas70.addData('response_vas70.duration', response_vas70.duration)
        # the Routine "feedback_vas70" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "estimate_vas70" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('estimate_vas70.started', globalClock.getTime())
        # Run 'Begin Routine' code from estimator_vas70
        trial = loop_vas70.thisN
        estimator_vas70.conduct_trial(response=response_vas70.keys,trial=trial)
        answer_vas70.setText(f"Ihre Antwort war: Nein." if response_vas70.keys == "n" else "Ihre Antwort war: Ja."
        )
        # keep track of which components have finished
        estimate_vas70Components = [answer_vas70]
        for thisComponent in estimate_vas70Components:
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
        
        # --- Run Routine "estimate_vas70" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *answer_vas70* updates
            
            # if answer_vas70 is starting this frame...
            if answer_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answer_vas70.frameNStart = frameN  # exact frame index
                answer_vas70.tStart = t  # local t and not account for scr refresh
                answer_vas70.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answer_vas70, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'answer_vas70.started')
                # update status
                answer_vas70.status = STARTED
                answer_vas70.setAutoDraw(True)
            
            # if answer_vas70 is active this frame...
            if answer_vas70.status == STARTED:
                # update params
                pass
            
            # if answer_vas70 is stopping this frame...
            if answer_vas70.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > answer_vas70.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    answer_vas70.tStop = t  # not accounting for scr refresh
                    answer_vas70.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'answer_vas70.stopped')
                    # update status
                    answer_vas70.status = FINISHED
                    answer_vas70.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in estimate_vas70Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "estimate_vas70" ---
        for thisComponent in estimate_vas70Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('estimate_vas70.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed trials_vas70 repeats of 'loop_vas70'
    
    
    # --- Prepare to start Routine "info_vas0_transition" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('info_vas0_transition.started', globalClock.getTime())
    # Run 'Begin Routine' code from init_estimator_vas0
    # Preparing estimator for VAS 0
    # slight overshoot to showcase the full range of possible temperatures
    temp_start_vas0 = estimator_vas70.get_estimate() - temp_start_vas0_minus
    
    estimator_vas0 = BayesianEstimatorVAS(
        vas_value=0,
        temp_start= temp_start_vas0,
        temp_std=temp_std_vas0,
        trials=trials_vas0)
    key_info_vas0.keys = []
    key_info_vas0.rt = []
    _key_info_vas0_allKeys = []
    # keep track of which components have finished
    info_vas0_transitionComponents = [text_info_vas0, key_info_vas0]
    for thisComponent in info_vas0_transitionComponents:
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
    
    # --- Run Routine "info_vas0_transition" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_info_vas0* updates
        
        # if text_info_vas0 is starting this frame...
        if text_info_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_info_vas0.frameNStart = frameN  # exact frame index
            text_info_vas0.tStart = t  # local t and not account for scr refresh
            text_info_vas0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_info_vas0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_info_vas0.started')
            # update status
            text_info_vas0.status = STARTED
            text_info_vas0.setAutoDraw(True)
        
        # if text_info_vas0 is active this frame...
        if text_info_vas0.status == STARTED:
            # update params
            pass
        
        # *key_info_vas0* updates
        waitOnFlip = False
        
        # if key_info_vas0 is starting this frame...
        if key_info_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_info_vas0.frameNStart = frameN  # exact frame index
            key_info_vas0.tStart = t  # local t and not account for scr refresh
            key_info_vas0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_info_vas0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_info_vas0.started')
            # update status
            key_info_vas0.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_info_vas0.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_info_vas0.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_info_vas0.status == STARTED and not waitOnFlip:
            theseKeys = key_info_vas0.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_info_vas0_allKeys.extend(theseKeys)
            if len(_key_info_vas0_allKeys):
                key_info_vas0.keys = _key_info_vas0_allKeys[-1].name  # just the last key pressed
                key_info_vas0.rt = _key_info_vas0_allKeys[-1].rt
                key_info_vas0.duration = _key_info_vas0_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in info_vas0_transitionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "info_vas0_transition" ---
    for thisComponent in info_vas0_transitionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('info_vas0_transition.stopped', globalClock.getTime())
    # check responses
    if key_info_vas0.keys in ['', [], None]:  # No response was made
        key_info_vas0.keys = None
    thisExp.addData('key_info_vas0.keys',key_info_vas0.keys)
    if key_info_vas0.keys != None:  # we had a response
        thisExp.addData('key_info_vas0.rt', key_info_vas0.rt)
        thisExp.addData('key_info_vas0.duration', key_info_vas0.duration)
    thisExp.nextEntry()
    # the Routine "info_vas0_transition" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_vas0 = data.TrialHandler(nReps=trials_vas0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_vas0')
    thisExp.addLoop(loop_vas0)  # add the loop to the experiment
    thisLoop_vas0 = loop_vas0.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_vas0.rgb)
    if thisLoop_vas0 != None:
        for paramName in thisLoop_vas0:
            globals()[paramName] = thisLoop_vas0[paramName]
    
    for thisLoop_vas0 in loop_vas0:
        currentLoop = loop_vas0
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_vas0.rgb)
        if thisLoop_vas0 != None:
            for paramName in thisLoop_vas0:
                globals()[paramName] = thisLoop_vas0[paramName]
        
        # --- Prepare to start Routine "iti" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('iti.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (loop_preexposure.thisN == 0)
        # keep track of which components have finished
        itiComponents = [cross_neutral]
        for thisComponent in itiComponents:
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
        
        # --- Run Routine "iti" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_neutral* updates
            
            # if cross_neutral is starting this frame...
            if cross_neutral.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cross_neutral.frameNStart = frameN  # exact frame index
                cross_neutral.tStart = t  # local t and not account for scr refresh
                cross_neutral.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_neutral, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_neutral.started')
                # update status
                cross_neutral.status = STARTED
                cross_neutral.setAutoDraw(True)
            
            # if cross_neutral is active this frame...
            if cross_neutral.status == STARTED:
                # update params
                pass
            
            # if cross_neutral is stopping this frame...
            if cross_neutral.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_neutral.tStartRefresh + iti_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_neutral.tStop = t  # not accounting for scr refresh
                    cross_neutral.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_neutral.stopped')
                    # update status
                    cross_neutral.status = FINISHED
                    cross_neutral.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in itiComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in itiComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('iti.stopped', globalClock.getTime())
        # the Routine "iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_vas0" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_vas0.started', globalClock.getTime())
        # Run 'Begin Routine' code from thermoino_vas0
        checked = False
        stimuli_clock.reset()
        
        luigi.trigger()
        time_for_ramp_up = luigi.set_temp(estimator_vas0.current_temp)[1]
        # keep track of which components have finished
        trial_vas0Components = [cross_pain_vas0]
        for thisComponent in trial_vas0Components:
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
        
        # --- Run Routine "trial_vas0" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from thermoino_vas0
            if not checked:
                if stimuli_clock.getTime() > (time_for_ramp_up + stimuli_duration):
                    luigi.set_temp(mms_baseline)
                    checked = True
            
            # *cross_pain_vas0* updates
            
            # if cross_pain_vas0 is starting this frame...
            if cross_pain_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_pain_vas0.frameNStart = frameN  # exact frame index
                cross_pain_vas0.tStart = t  # local t and not account for scr refresh
                cross_pain_vas0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_pain_vas0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_pain_vas0.started')
                # update status
                cross_pain_vas0.status = STARTED
                cross_pain_vas0.setAutoDraw(True)
            
            # if cross_pain_vas0 is active this frame...
            if cross_pain_vas0.status == STARTED:
                # update params
                pass
            
            # if cross_pain_vas0 is stopping this frame...
            if cross_pain_vas0.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_pain_vas0.tStartRefresh + stimuli_duration + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_pain_vas0.tStop = t  # not accounting for scr refresh
                    cross_pain_vas0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_pain_vas0.stopped')
                    # update status
                    cross_pain_vas0.status = FINISHED
                    cross_pain_vas0.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_vas0Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_vas0" ---
        for thisComponent in trial_vas0Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_vas0.stopped', globalClock.getTime())
        # the Routine "trial_vas0" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_vas0" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback_vas0.started', globalClock.getTime())
        response_vas0.keys = []
        response_vas0.rt = []
        _response_vas0_allKeys = []
        # keep track of which components have finished
        feedback_vas0Components = [question_vas0, response_vas0]
        for thisComponent in feedback_vas0Components:
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
        
        # --- Run Routine "feedback_vas0" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_vas0* updates
            
            # if question_vas0 is starting this frame...
            if question_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_vas0.frameNStart = frameN  # exact frame index
                question_vas0.tStart = t  # local t and not account for scr refresh
                question_vas0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_vas0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_vas0.started')
                # update status
                question_vas0.status = STARTED
                question_vas0.setAutoDraw(True)
            
            # if question_vas0 is active this frame...
            if question_vas0.status == STARTED:
                # update params
                pass
            
            # if question_vas0 is stopping this frame...
            if question_vas0.status == STARTED:
                if bool(response_vas0.status==FINISHED):
                    # keep track of stop time/frame for later
                    question_vas0.tStop = t  # not accounting for scr refresh
                    question_vas0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_vas0.stopped')
                    # update status
                    question_vas0.status = FINISHED
                    question_vas0.setAutoDraw(False)
            
            # *response_vas0* updates
            waitOnFlip = False
            
            # if response_vas0 is starting this frame...
            if response_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                response_vas0.frameNStart = frameN  # exact frame index
                response_vas0.tStart = t  # local t and not account for scr refresh
                response_vas0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(response_vas0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'response_vas0.started')
                # update status
                response_vas0.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(response_vas0.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(response_vas0.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if response_vas0.status == STARTED and not waitOnFlip:
                theseKeys = response_vas0.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=True)
                _response_vas0_allKeys.extend(theseKeys)
                if len(_response_vas0_allKeys):
                    response_vas0.keys = _response_vas0_allKeys[-1].name  # just the last key pressed
                    response_vas0.rt = _response_vas0_allKeys[-1].rt
                    response_vas0.duration = _response_vas0_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_vas0Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_vas0" ---
        for thisComponent in feedback_vas0Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback_vas0.stopped', globalClock.getTime())
        # check responses
        if response_vas0.keys in ['', [], None]:  # No response was made
            response_vas0.keys = None
        loop_vas0.addData('response_vas0.keys',response_vas0.keys)
        if response_vas0.keys != None:  # we had a response
            loop_vas0.addData('response_vas0.rt', response_vas0.rt)
            loop_vas0.addData('response_vas0.duration', response_vas0.duration)
        # the Routine "feedback_vas0" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "estimate_vas0" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('estimate_vas0.started', globalClock.getTime())
        # Run 'Begin Routine' code from estimator_vas0
        trial = loop_vas0.thisN
        estimator_vas0.conduct_trial(response=response_vas0.keys,trial=trial)
        answer_vas0.setText(f"Ihre Antwort war: Nein." if response_vas0.keys == "n" else "Ihre Antwort war: Ja."
        )
        # keep track of which components have finished
        estimate_vas0Components = [answer_vas0]
        for thisComponent in estimate_vas0Components:
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
        
        # --- Run Routine "estimate_vas0" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *answer_vas0* updates
            
            # if answer_vas0 is starting this frame...
            if answer_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answer_vas0.frameNStart = frameN  # exact frame index
                answer_vas0.tStart = t  # local t and not account for scr refresh
                answer_vas0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answer_vas0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'answer_vas0.started')
                # update status
                answer_vas0.status = STARTED
                answer_vas0.setAutoDraw(True)
            
            # if answer_vas0 is active this frame...
            if answer_vas0.status == STARTED:
                # update params
                pass
            
            # if answer_vas0 is stopping this frame...
            if answer_vas0.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > answer_vas0.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    answer_vas0.tStop = t  # not accounting for scr refresh
                    answer_vas0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'answer_vas0.stopped')
                    # update status
                    answer_vas0.status = FINISHED
                    answer_vas0.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in estimate_vas0Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "estimate_vas0" ---
        for thisComponent in estimate_vas0Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('estimate_vas0.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed trials_vas0 repeats of 'loop_vas0'
    
    
    # --- Prepare to start Routine "bye" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('bye.started', globalClock.getTime())
    key_bye.keys = []
    key_bye.rt = []
    _key_bye_allKeys = []
    # keep track of which components have finished
    byeComponents = [text_bye, key_bye]
    for thisComponent in byeComponents:
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
    
    # --- Run Routine "bye" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_bye* updates
        
        # if text_bye is starting this frame...
        if text_bye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_bye.frameNStart = frameN  # exact frame index
            text_bye.tStart = t  # local t and not account for scr refresh
            text_bye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_bye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_bye.started')
            # update status
            text_bye.status = STARTED
            text_bye.setAutoDraw(True)
        
        # if text_bye is active this frame...
        if text_bye.status == STARTED:
            # update params
            pass
        
        # *key_bye* updates
        waitOnFlip = False
        
        # if key_bye is starting this frame...
        if key_bye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_bye.frameNStart = frameN  # exact frame index
            key_bye.tStart = t  # local t and not account for scr refresh
            key_bye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_bye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_bye.started')
            # update status
            key_bye.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_bye.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_bye.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_bye.status == STARTED and not waitOnFlip:
            theseKeys = key_bye.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_bye_allKeys.extend(theseKeys)
            if len(_key_bye_allKeys):
                key_bye.keys = _key_bye_allKeys[-1].name  # just the last key pressed
                key_bye.rt = _key_bye_allKeys[-1].rt
                key_bye.duration = _key_bye_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in byeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "bye" ---
    for thisComponent in byeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('bye.stopped', globalClock.getTime())
    # Run 'End Routine' code from save_participant_data
    add_participant(
        expInfo['participant'],
        expInfo['age'],
        expInfo['gender'],
        estimator_vas0.get_estimate(),
        estimator_vas70.get_estimate())
    # check responses
    if key_bye.keys in ['', [], None]:  # No response was made
        key_bye.keys = None
    thisExp.addData('key_bye.keys',key_bye.keys)
    if key_bye.keys != None:  # we had a response
        thisExp.addData('key_bye.rt', key_bye.rt)
        thisExp.addData('key_bye.duration', key_bye.duration)
    thisExp.nextEntry()
    # the Routine "bye" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from all_variables
    close_root_logging()
    # Run 'End Experiment' code from thermoino
    luigi.close()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            eyetracker.setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
