#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on September 15, 2023, at 12:59
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

# Run 'Before Experiment' code from all_variables
from src.experiments.logger import setup_logger, close_logger
logger_for_runner = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)

# Thermoino
port = "COM7" # COM7 for top usb port on the front, use list_com_ports() to find out
temp_baseline = 30 # has to be the same as in MMS
rate_of_rise = 5 # has to be the same as in MMS

# Stimuli
stimuli_clock = core.Clock()
stimuli_duration = 0.2 # 8
iti_duration = 0.2 # 8  + np.random.randint(0, 5)

# Pre-exposure
temps_preexposure = [35, 36, 37]

# Estimator
trials_vas0 = 7 # is the same as nReps in psychopy loop
temp_start_vas0 = 39.
temp_std_vas0 = 3.5

trials_vas70 = 5 # is the same as nReps in psychopy loop
temp_start_vas70 = None #  will be set after VAS 0 estimate
temp_start_vas70_plus = 3 # VAS 0 estimate plus int
temp_std_vas70 = 1.5 # smaller std for higher temperatures

# Run 'Before Experiment' code from thermoino
from src.experiments.thermoino import Thermoino
# Run 'Before Experiment' code from estimator_vas0
from src.experiments.calibration import BayesianEstimatorVAS

# Estimator for VAS 0 (pain threshold)
estimator_vas0 = BayesianEstimatorVAS(
    vas_value= 0, 
    temp_start=temp_start_vas0,
    temp_std=temp_std_vas0,
    trials=trials_vas0)

# Run 'Before Experiment' code from estimator_vas70
# instantiating here does not make sense because we need the values from the VAS 0 esitmator first
# Run 'Before Experiment' code from save_participant_data
from src.experiments.participant_data import add_participant
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'calibration_exp'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'age': '0',
    'gender': '### Female or Male ###',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


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
        originPath='G:\\Meine Ablage\\PhD\\Code\\mpad-pilot\\experiments\\calibration\\calibration_exp.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
            units='height'
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
        win.units = 'height'
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
        temp_baseline=temp_baseline, # has to be the same as in MMS
        rate_of_rise=rate_of_rise) # has to be the same as in MMS
    
    luigi.connect()
    luigi.trigger()
    
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Willkommen zur Studie!\n\nDie Studie besteht aus zwei Teilen:\n1. einer Schmerz-Kalibrierung und\n2. einem Hauptexperiment.\n\nWir beginnen mit dem ersten Teil, der Schmerz-Kalibrierung.\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "explanation_preexposure" ---
    text_explanation_preexposure = visual.TextStim(win=win, name='text_explanation_preexposure',
        text='Die Schmerz-Kalibrierung besteht aus 3 Teilen:\n\n1. Aufwärmung der Hautstelle,\n2. Bestimmung, ab wann Sie Schmerz spüren,\n3. Bestimmung, ab wann Sie starken Schmerz spüren.\n\nWir beginnen mit dem Aufwärmen der Hautstelle.\nHierbei müssen Sie nichts weiter tun.\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_explanation_preexposure = keyboard.Keyboard()
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "preexposure" ---
    corss_pain_preexposure = visual.ShapeStim(
        win=win, name='corss_pain_preexposure', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_preexposure" ---
    question_preexposure = visual.TextStim(win=win, name='question_preexposure',
        text='War einer der Reize schmerzhaft?\n\n(y/n)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    response_preexposure = keyboard.Keyboard()
    
    # --- Initialize components for Routine "explanation_vas0" ---
    text_explanation_vas0 = visual.TextStim(win=win, name='text_explanation_vas0',
        text='Wunderbar!\n\n\nEs beginnt nun der zweite Teil der Schmerzkalibrierung.\n\nDabei möchten wir die Schwelle bestimmen, ab wann Sie Schmerzen spüren.\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_explanation_vas0 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_vas0" ---
    cross_pain_vas0 = visual.ShapeStim(
        win=win, name='cross_pain_vas0', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_vas0" ---
    question_vas0 = visual.TextStim(win=win, name='question_vas0',
        text='War der Reiz schmerzhaft?\n\n(y/n)',
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
        depth=0.0);
    
    # --- Initialize components for Routine "transition_vas0_to_vas70" ---
    text_explanation_vas70 = visual.TextStim(win=win, name='text_explanation_vas70',
        text='Als nächstes erhalten Sie einen starken Schmerzreiz.\nAuf einer Skala von 1 bis 10 soll er eine 7 darstellen.\n\n1: kein Schmerz,                           \n10: unerträglicher Schmerz,\n7: schwerer Schmerz .                   \n\n(Bild einfügen)\n\n\n--- Leertaste zum Forfahren ---',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_explanation_vas70 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "explanation_vas70" ---
    
    # --- Initialize components for Routine "iti" ---
    cross_neutral = visual.ShapeStim(
        win=win, name='cross_neutral', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_vas70" ---
    cross_pain_vas70 = visual.ShapeStim(
        win=win, name='cross_pain_vas70', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "feedback_vas70" ---
    question_vas70 = visual.TextStim(win=win, name='question_vas70',
        text='War der Reiz eine 7 von 10 auf der Schmerzskala?\n\n(y/n)',
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
        depth=0.0);
    
    # --- Initialize components for Routine "bye" ---
    text_bye = visual.TextStim(win=win, name='text_bye',
        text='Thanks!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # --- Prepare to start Routine "explanation_preexposure" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('explanation_preexposure.started', globalClock.getTime())
    key_explanation_preexposure.keys = []
    key_explanation_preexposure.rt = []
    _key_explanation_preexposure_allKeys = []
    # keep track of which components have finished
    explanation_preexposureComponents = [text_explanation_preexposure, key_explanation_preexposure]
    for thisComponent in explanation_preexposureComponents:
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
    
    # --- Run Routine "explanation_preexposure" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explanation_preexposure* updates
        
        # if text_explanation_preexposure is starting this frame...
        if text_explanation_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explanation_preexposure.frameNStart = frameN  # exact frame index
            text_explanation_preexposure.tStart = t  # local t and not account for scr refresh
            text_explanation_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explanation_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explanation_preexposure.started')
            # update status
            text_explanation_preexposure.status = STARTED
            text_explanation_preexposure.setAutoDraw(True)
        
        # if text_explanation_preexposure is active this frame...
        if text_explanation_preexposure.status == STARTED:
            # update params
            pass
        
        # *key_explanation_preexposure* updates
        waitOnFlip = False
        
        # if key_explanation_preexposure is starting this frame...
        if key_explanation_preexposure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_explanation_preexposure.frameNStart = frameN  # exact frame index
            key_explanation_preexposure.tStart = t  # local t and not account for scr refresh
            key_explanation_preexposure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explanation_preexposure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explanation_preexposure.started')
            # update status
            key_explanation_preexposure.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explanation_preexposure.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explanation_preexposure.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explanation_preexposure.status == STARTED and not waitOnFlip:
            theseKeys = key_explanation_preexposure.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explanation_preexposure_allKeys.extend(theseKeys)
            if len(_key_explanation_preexposure_allKeys):
                key_explanation_preexposure.keys = _key_explanation_preexposure_allKeys[-1].name  # just the last key pressed
                key_explanation_preexposure.rt = _key_explanation_preexposure_allKeys[-1].rt
                key_explanation_preexposure.duration = _key_explanation_preexposure_allKeys[-1].duration
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
        for thisComponent in explanation_preexposureComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "explanation_preexposure" ---
    for thisComponent in explanation_preexposureComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('explanation_preexposure.stopped', globalClock.getTime())
    # check responses
    if key_explanation_preexposure.keys in ['', [], None]:  # No response was made
        key_explanation_preexposure.keys = None
    thisExp.addData('key_explanation_preexposure.keys',key_explanation_preexposure.keys)
    if key_explanation_preexposure.keys != None:  # we had a response
        thisExp.addData('key_explanation_preexposure.rt', key_explanation_preexposure.rt)
        thisExp.addData('key_explanation_preexposure.duration', key_explanation_preexposure.duration)
    thisExp.nextEntry()
    # the Routine "explanation_preexposure" was not non-slip safe, so reset the non-slip timer
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
        # Run 'Begin Routine' code from prexposure
        checked = False
        stimuli_clock.reset()
        
        trial = loop_preexposure.thisN
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
            # Run 'Each Frame' code from prexposure
            routine_duration = time_for_ramp_up + stimuli_clock.getTime() 
            
            if not checked:
                if routine_duration > stimuli_duration:
                    luigi.set_temp(temp_baseline)
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
    # Run 'End Routine' code from transition_preexposure_to_vas0
    # If response was yes
    logger_for_runner.info("Preexposure painful? Answer: %s", response_preexposure.keys)
    
    if response_preexposure.keys == "y":
        # Decrease starting temperature
        global temp_start_vas0 # psychopy has to be able to find it in the spaghetti
        temp_start_vas0 -= 2
        # Reinitialize estimator for VAS 0 with different temp_start
        global estimator_vas0 
        estimator_vas0 = BayesianEstimatorVAS(
            vas_value=0,
            temp_start=temp_start_vas0,
            temp_std=temp_std_vas0,
            trials=trials_vas0)
    # the Routine "feedback_preexposure" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "explanation_vas0" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('explanation_vas0.started', globalClock.getTime())
    key_explanation_vas0.keys = []
    key_explanation_vas0.rt = []
    _key_explanation_vas0_allKeys = []
    # keep track of which components have finished
    explanation_vas0Components = [text_explanation_vas0, key_explanation_vas0]
    for thisComponent in explanation_vas0Components:
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
    
    # --- Run Routine "explanation_vas0" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explanation_vas0* updates
        
        # if text_explanation_vas0 is starting this frame...
        if text_explanation_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explanation_vas0.frameNStart = frameN  # exact frame index
            text_explanation_vas0.tStart = t  # local t and not account for scr refresh
            text_explanation_vas0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explanation_vas0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explanation_vas0.started')
            # update status
            text_explanation_vas0.status = STARTED
            text_explanation_vas0.setAutoDraw(True)
        
        # if text_explanation_vas0 is active this frame...
        if text_explanation_vas0.status == STARTED:
            # update params
            pass
        
        # if text_explanation_vas0 is stopping this frame...
        if text_explanation_vas0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_explanation_vas0.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_explanation_vas0.tStop = t  # not accounting for scr refresh
                text_explanation_vas0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_explanation_vas0.stopped')
                # update status
                text_explanation_vas0.status = FINISHED
                text_explanation_vas0.setAutoDraw(False)
        
        # *key_explanation_vas0* updates
        waitOnFlip = False
        
        # if key_explanation_vas0 is starting this frame...
        if key_explanation_vas0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_explanation_vas0.frameNStart = frameN  # exact frame index
            key_explanation_vas0.tStart = t  # local t and not account for scr refresh
            key_explanation_vas0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explanation_vas0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explanation_vas0.started')
            # update status
            key_explanation_vas0.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explanation_vas0.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explanation_vas0.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explanation_vas0.status == STARTED and not waitOnFlip:
            theseKeys = key_explanation_vas0.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explanation_vas0_allKeys.extend(theseKeys)
            if len(_key_explanation_vas0_allKeys):
                key_explanation_vas0.keys = _key_explanation_vas0_allKeys[-1].name  # just the last key pressed
                key_explanation_vas0.rt = _key_explanation_vas0_allKeys[-1].rt
                key_explanation_vas0.duration = _key_explanation_vas0_allKeys[-1].duration
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
        for thisComponent in explanation_vas0Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "explanation_vas0" ---
    for thisComponent in explanation_vas0Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('explanation_vas0.stopped', globalClock.getTime())
    # check responses
    if key_explanation_vas0.keys in ['', [], None]:  # No response was made
        key_explanation_vas0.keys = None
    thisExp.addData('key_explanation_vas0.keys',key_explanation_vas0.keys)
    if key_explanation_vas0.keys != None:  # we had a response
        thisExp.addData('key_explanation_vas0.rt', key_explanation_vas0.rt)
        thisExp.addData('key_explanation_vas0.duration', key_explanation_vas0.duration)
    thisExp.nextEntry()
    # the Routine "explanation_vas0" was not non-slip safe, so reset the non-slip timer
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
            # Run 'Each Frame' code from thermoino_vas0
            routine_duration = time_for_ramp_up + stimuli_clock.getTime() 
            
            if not checked:
                if routine_duration > stimuli_duration:
                    luigi.set_temp(temp_baseline)
                    checked = True
            
            
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
        answer_vas0.setText(f"Ihre Antwort war {response_vas0.keys}.")
        # Run 'Begin Routine' code from estimator_vas0
        trial = loop_vas0.thisN
        estimator_vas0.conduct_trial(response=response_vas0.keys,trial=trial)
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
    
    
    # --- Prepare to start Routine "transition_vas0_to_vas70" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('transition_vas0_to_vas70.started', globalClock.getTime())
    # Run 'Begin Routine' code from init_estimator_vas70
    # Preparing estimator for VAS 70
    # slight overshoot to showcase the full range of possible temperatures
    temp_start_vas70 = estimator_vas0.get_estimate() + temp_start_vas70_plus 
    
    estimator_vas70 = BayesianEstimatorVAS(
        vas_value=70,
        temp_start= temp_start_vas70,
        temp_std=temp_std_vas70,
        trials=trials_vas70)
    key_explanation_vas70.keys = []
    key_explanation_vas70.rt = []
    _key_explanation_vas70_allKeys = []
    # keep track of which components have finished
    transition_vas0_to_vas70Components = [text_explanation_vas70, key_explanation_vas70]
    for thisComponent in transition_vas0_to_vas70Components:
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
    
    # --- Run Routine "transition_vas0_to_vas70" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explanation_vas70* updates
        
        # if text_explanation_vas70 is starting this frame...
        if text_explanation_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explanation_vas70.frameNStart = frameN  # exact frame index
            text_explanation_vas70.tStart = t  # local t and not account for scr refresh
            text_explanation_vas70.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explanation_vas70, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explanation_vas70.started')
            # update status
            text_explanation_vas70.status = STARTED
            text_explanation_vas70.setAutoDraw(True)
        
        # if text_explanation_vas70 is active this frame...
        if text_explanation_vas70.status == STARTED:
            # update params
            pass
        
        # *key_explanation_vas70* updates
        waitOnFlip = False
        
        # if key_explanation_vas70 is starting this frame...
        if key_explanation_vas70.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_explanation_vas70.frameNStart = frameN  # exact frame index
            key_explanation_vas70.tStart = t  # local t and not account for scr refresh
            key_explanation_vas70.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explanation_vas70, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explanation_vas70.started')
            # update status
            key_explanation_vas70.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explanation_vas70.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explanation_vas70.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explanation_vas70.status == STARTED and not waitOnFlip:
            theseKeys = key_explanation_vas70.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explanation_vas70_allKeys.extend(theseKeys)
            if len(_key_explanation_vas70_allKeys):
                key_explanation_vas70.keys = _key_explanation_vas70_allKeys[-1].name  # just the last key pressed
                key_explanation_vas70.rt = _key_explanation_vas70_allKeys[-1].rt
                key_explanation_vas70.duration = _key_explanation_vas70_allKeys[-1].duration
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
        for thisComponent in transition_vas0_to_vas70Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "transition_vas0_to_vas70" ---
    for thisComponent in transition_vas0_to_vas70Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('transition_vas0_to_vas70.stopped', globalClock.getTime())
    # check responses
    if key_explanation_vas70.keys in ['', [], None]:  # No response was made
        key_explanation_vas70.keys = None
    thisExp.addData('key_explanation_vas70.keys',key_explanation_vas70.keys)
    if key_explanation_vas70.keys != None:  # we had a response
        thisExp.addData('key_explanation_vas70.rt', key_explanation_vas70.rt)
        thisExp.addData('key_explanation_vas70.duration', key_explanation_vas70.duration)
    thisExp.nextEntry()
    # the Routine "transition_vas0_to_vas70" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "explanation_vas70" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('explanation_vas70.started', globalClock.getTime())
    # keep track of which components have finished
    explanation_vas70Components = []
    for thisComponent in explanation_vas70Components:
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
    
    # --- Run Routine "explanation_vas70" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
        for thisComponent in explanation_vas70Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "explanation_vas70" ---
    for thisComponent in explanation_vas70Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('explanation_vas70.stopped', globalClock.getTime())
    # the Routine "explanation_vas70" was not non-slip safe, so reset the non-slip timer
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
            # Run 'Each Frame' code from thermoino_vas70
            routine_duration = time_for_ramp_up + stimuli_clock.getTime()
            
            if not checked:
                if routine_duration > stimuli_duration:
                    luigi.set_temp(temp_baseline)
                    checked = True
            
            
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
        answer_vas70.setText(f"Ihre Antwort war {response_vas70.keys}.")
        # Run 'Begin Routine' code from estimator_vas70
        trial = loop_vas70.thisN
        estimator_vas70.conduct_trial(response=response_vas70.keys,trial=trial)
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
    
    
    # --- Prepare to start Routine "bye" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('bye.started', globalClock.getTime())
    # keep track of which components have finished
    byeComponents = [text_bye]
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
    while continueRoutine and routineTimer.getTime() < 1.0:
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
        
        # if text_bye is stopping this frame...
        if text_bye.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_bye.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_bye.tStop = t  # not accounting for scr refresh
                text_bye.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_bye.stopped')
                # update status
                text_bye.status = FINISHED
                text_bye.setAutoDraw(False)
        
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
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    # Run 'End Experiment' code from all_variables
    close_logger(logger_for_runner)
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
            inputs['eyetracker'].setConnectionState(False)
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
