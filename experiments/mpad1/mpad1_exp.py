#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.1),
    on September 18, 2023, at 11:22
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
expName = 'mpad1_exp'  # from the Builder filename that created this script
expInfo = {
    '': '',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Run 'Before Experiment' code from all_variables
# TODO
# fix baseline_temp & temp_baseline for all scripts
# set start_study_mode to NoPrompt

# Root logger
from pathlib import Path
from datetime import datetime
from src.experiments.logger import setup_logger, close_logger

log_dir = Path('log')
log_dir.mkdir(parents=True, exist_ok=True)
log_filename_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
log_file = log_dir / log_filename_str

psychopy_logger = setup_logger(
    '',
    level=logging.INFO, 
    log_file=log_file,
    stream_handler=False)

# Info
from src.experiments.participant_data import read_last_participant
participant_info = read_last_participant()
expInfo['expName'] = "mpad1_exp"
expInfo['participant'] = participant_info['participant']
expInfo['age'] = participant_info['age']
expInfo['gender'] = participant_info['gender']

# iMotions
start_study_mode = "NormalPrompt"

# Stimuli
seeds = [463]#, 320]#, 43, 999, 242, 32, 1, 98, 478, 48, 435]
n_trials = len(seeds)
this_trial = None # just initializing

minimal_desired_duration = 2 # in seconds
periods = [67, 20] # [0] is the baseline and [1] the modulation; in seconds
frequencies = 1./np.array(periods)
# calculate amplitudes based on VAS 70 - VAS 0
temp_range = participant_info['temp_range'] # VAS 70 - VAS 0
amplitudes = [1/3*temp_range/2, 2/3*temp_range/2]

sample_rate = 60
random_periods = True
baseline_temp = participant_info['baseline_temp'] # @ VAS 35

# While debugging, the code in stimuli_function is outcommented
plateau_duration = 20
n_plateaus = 4
add_at_start = "random"
add_at_end = True

# Time
stimuli_clock = core.Clock()
iti_duration = 2 # 8  + np.random.randint(0, 5)
vas_labels = ("Keine\nSchmerzen", "Sehr starke Schmerzen")

# Thermoino
port = "COM7" # use list_com_ports() beforehand to find out
temp_baseline = 30 # has to be the same as in MMS (not the same as baseline_temp (sorry for confusing names))
rate_of_rise = 5 # has to be the same as in MMS
bin_size_ms = 500
# Run 'Before Experiment' code from imotions_control
from src.experiments.imotions import RemoteControliMotions
# Run 'Before Experiment' code from mouse_instruction
import src.experiments.mouse_action as mouse_action
from src.experiments.mouse_action import pixel_pos_y
# Run 'Before Experiment' code from stimuli_function
from src.experiments.stimuli_function import StimuliFunction
# Run 'Before Experiment' code from thermoino
from src.experiments.thermoino import ThermoinoComplexTimeCourses
# Run 'Before Experiment' code from imotions_event
from src.experiments.imotions import EventRecievingiMotions
from psychopy import core


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
        originPath='G:\\Meine Ablage\\PhD\\Code\\mpad-pilot\\experiments\\mpad1\\mpad1_exp.py',
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
            size=[900, 600], fullscr=False, screen=0,
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
    # Run 'Begin Experiment' code from imotions_control
    imotions_control = RemoteControliMotions(
        study = expInfo['expName'],
        participant = expInfo['participant'],
        age = expInfo['age'],
        gender = expInfo['gender']
    )
    
    imotions_control.connect()
    imotions_control.start_study(mode=start_study_mode)
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Willkommen zum Haupt-Experiment!\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_2" ---
    text_welcome_2 = visual.TextStim(win=win, name='text_welcome_2',
        text='Mit diesem Experiment möchten wir Schmerz objektiv messbar machen.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_3" ---
    text_welcome_3 = visual.TextStim(win=win, name='text_welcome_3',
        text='Das Experiment besteht aus insgesamt 12 Blöcken á 4 Minuten.\n\nIn jedem Block erhalten Sie Temperatur-Reize mit schwankender Intensität.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instruction" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Ihre Aufgabe ist es, Ihren Schmerzpegel fortwährend zu bewerten.\nDafür bewegen Sie den Mauszeiger auf der Skala einfach hin und her.\n\n\n\n\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruction = keyboard.Keyboard()
    vas_cont_instruction = visual.Slider(win=win, name='vas_cont_instruction',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "instruction_2" ---
    text_instruction_2 = visual.TextStim(win=win, name='text_instruction_2',
        text='Die Beschriftungen "Keine Schmerzen" und "Sehr starke Schmerzen" entsprechen dabei Ihren Werten aus der Kalibrierung. Nutzen Sie die gesamte Breite der Skala!\n\n\n\n\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruction_2 = keyboard.Keyboard()
    vas_cont_instruction_2 = visual.Slider(win=win, name='vas_cont_instruction_2',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "ready" ---
    text_ready = visual.TextStim(win=win, name='text_ready',
        text='Sollten Sie noch Fragen haben, richten Sie sich bitte jetzt an die Versuchsleitung.\n\nAnsonsten startet nun das Experiment.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_ready = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_prep" ---
    vas_cont_prep = visual.Slider(win=win, name='vas_cont_prep',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # --- Initialize components for Routine "trial_vas_continuous" ---
    # Run 'Begin Experiment' code from thermoino
    luigi = ThermoinoComplexTimeCourses(
        port=port, 
        mms_temp_baseline=temp_baseline, 
        mms_rate_of_rise=rate_of_rise)
    
    luigi.connect()
    # Run 'Begin Experiment' code from imotions_event
    """ Connect with event recieving API """
    imotions_event = EventRecievingiMotions()
    imotions_event.connect()
    imotions_event.start_study
    
    # create a clock
    stimuli_clock = core.Clock()
    vas_cont = visual.Slider(win=win, name='vas_cont',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # --- Initialize components for Routine "trial_end" ---
    # Run 'Begin Experiment' code from trial_end
    this_trial = None
    text_trial_end = visual.TextStim(win=win, name='text_trial_end',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_trial_end = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_next" ---
    key_trial_next = keyboard.Keyboard()
    text_trial_next = visual.TextStim(win=win, name='text_trial_next',
        text='(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
            theseKeys = key_welcome.getKeys(keyList=['j','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
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
    
    # --- Prepare to start Routine "instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction.started', globalClock.getTime())
    # Run 'Begin Routine' code from mouse_instruction
    vas_pos_y = pixel_pos_y(
        component_pos = vas_cont.pos,
        win_size = win.size, 
        win_pos = win.pos)
        
    mouse_action.hold()
    key_instruction.keys = []
    key_instruction.rt = []
    _key_instruction_allKeys = []
    vas_cont_instruction.reset()
    # keep track of which components have finished
    instructionComponents = [text_instruction, key_instruction, vas_cont_instruction]
    for thisComponent in instructionComponents:
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
    
    # --- Run Routine "instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from mouse_instruction
        mouse_action.check(vas_pos_y)
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # *key_instruction* updates
        waitOnFlip = False
        
        # if key_instruction is starting this frame...
        if key_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruction.frameNStart = frameN  # exact frame index
            key_instruction.tStart = t  # local t and not account for scr refresh
            key_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruction.started')
            # update status
            key_instruction.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruction.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruction.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruction.status == STARTED and not waitOnFlip:
            theseKeys = key_instruction.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruction_allKeys.extend(theseKeys)
            if len(_key_instruction_allKeys):
                key_instruction.keys = _key_instruction_allKeys[-1].name  # just the last key pressed
                key_instruction.rt = _key_instruction_allKeys[-1].rt
                key_instruction.duration = _key_instruction_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *vas_cont_instruction* updates
        
        # if vas_cont_instruction is starting this frame...
        if vas_cont_instruction.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            vas_cont_instruction.frameNStart = frameN  # exact frame index
            vas_cont_instruction.tStart = t  # local t and not account for scr refresh
            vas_cont_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vas_cont_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vas_cont_instruction.started')
            # update status
            vas_cont_instruction.status = STARTED
            vas_cont_instruction.setAutoDraw(True)
        
        # if vas_cont_instruction is active this frame...
        if vas_cont_instruction.status == STARTED:
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
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction.stopped', globalClock.getTime())
    # check responses
    if key_instruction.keys in ['', [], None]:  # No response was made
        key_instruction.keys = None
    thisExp.addData('key_instruction.keys',key_instruction.keys)
    if key_instruction.keys != None:  # we had a response
        thisExp.addData('key_instruction.rt', key_instruction.rt)
        thisExp.addData('key_instruction.duration', key_instruction.duration)
    thisExp.nextEntry()
    thisExp.addData('vas_cont_instruction.response', vas_cont_instruction.getRating())
    thisExp.addData('vas_cont_instruction.rt', vas_cont_instruction.getRT())
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_2.started', globalClock.getTime())
    key_instruction_2.keys = []
    key_instruction_2.rt = []
    _key_instruction_2_allKeys = []
    vas_cont_instruction_2.reset()
    # keep track of which components have finished
    instruction_2Components = [text_instruction_2, key_instruction_2, vas_cont_instruction_2]
    for thisComponent in instruction_2Components:
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
    
    # --- Run Routine "instruction_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from mouse_instruction_2
        mouse_action.check(vas_pos_y)
        
        # *text_instruction_2* updates
        
        # if text_instruction_2 is starting this frame...
        if text_instruction_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction_2.frameNStart = frameN  # exact frame index
            text_instruction_2.tStart = t  # local t and not account for scr refresh
            text_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction_2.started')
            # update status
            text_instruction_2.status = STARTED
            text_instruction_2.setAutoDraw(True)
        
        # if text_instruction_2 is active this frame...
        if text_instruction_2.status == STARTED:
            # update params
            pass
        
        # *key_instruction_2* updates
        waitOnFlip = False
        
        # if key_instruction_2 is starting this frame...
        if key_instruction_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruction_2.frameNStart = frameN  # exact frame index
            key_instruction_2.tStart = t  # local t and not account for scr refresh
            key_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruction_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruction_2.started')
            # update status
            key_instruction_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruction_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruction_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruction_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruction_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruction_2_allKeys.extend(theseKeys)
            if len(_key_instruction_2_allKeys):
                key_instruction_2.keys = _key_instruction_2_allKeys[-1].name  # just the last key pressed
                key_instruction_2.rt = _key_instruction_2_allKeys[-1].rt
                key_instruction_2.duration = _key_instruction_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *vas_cont_instruction_2* updates
        
        # if vas_cont_instruction_2 is starting this frame...
        if vas_cont_instruction_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            vas_cont_instruction_2.frameNStart = frameN  # exact frame index
            vas_cont_instruction_2.tStart = t  # local t and not account for scr refresh
            vas_cont_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vas_cont_instruction_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vas_cont_instruction_2.started')
            # update status
            vas_cont_instruction_2.status = STARTED
            vas_cont_instruction_2.setAutoDraw(True)
        
        # if vas_cont_instruction_2 is active this frame...
        if vas_cont_instruction_2.status == STARTED:
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
        for thisComponent in instruction_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_2" ---
    for thisComponent in instruction_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction_2.stopped', globalClock.getTime())
    # Run 'End Routine' code from mouse_instruction_2
    mouse_action.release()
    # check responses
    if key_instruction_2.keys in ['', [], None]:  # No response was made
        key_instruction_2.keys = None
    thisExp.addData('key_instruction_2.keys',key_instruction_2.keys)
    if key_instruction_2.keys != None:  # we had a response
        thisExp.addData('key_instruction_2.rt', key_instruction_2.rt)
        thisExp.addData('key_instruction_2.duration', key_instruction_2.duration)
    thisExp.nextEntry()
    thisExp.addData('vas_cont_instruction_2.response', vas_cont_instruction_2.getRating())
    thisExp.addData('vas_cont_instruction_2.rt', vas_cont_instruction_2.getRT())
    # the Routine "instruction_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ready" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ready.started', globalClock.getTime())
    key_ready.keys = []
    key_ready.rt = []
    _key_ready_allKeys = []
    # keep track of which components have finished
    readyComponents = [text_ready, key_ready]
    for thisComponent in readyComponents:
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
    
    # --- Run Routine "ready" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ready* updates
        
        # if text_ready is starting this frame...
        if text_ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ready.frameNStart = frameN  # exact frame index
            text_ready.tStart = t  # local t and not account for scr refresh
            text_ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ready.started')
            # update status
            text_ready.status = STARTED
            text_ready.setAutoDraw(True)
        
        # if text_ready is active this frame...
        if text_ready.status == STARTED:
            # update params
            pass
        
        # *key_ready* updates
        waitOnFlip = False
        
        # if key_ready is starting this frame...
        if key_ready.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_ready.frameNStart = frameN  # exact frame index
            key_ready.tStart = t  # local t and not account for scr refresh
            key_ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_ready.started')
            # update status
            key_ready.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_ready.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_ready.status == STARTED and not waitOnFlip:
            theseKeys = key_ready.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_ready_allKeys.extend(theseKeys)
            if len(_key_ready_allKeys):
                key_ready.keys = _key_ready_allKeys[-1].name  # just the last key pressed
                key_ready.rt = _key_ready_allKeys[-1].rt
                key_ready.duration = _key_ready_allKeys[-1].duration
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
        for thisComponent in readyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ready" ---
    for thisComponent in readyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ready.stopped', globalClock.getTime())
    # check responses
    if key_ready.keys in ['', [], None]:  # No response was made
        key_ready.keys = None
    thisExp.addData('key_ready.keys',key_ready.keys)
    if key_ready.keys != None:  # we had a response
        thisExp.addData('key_ready.rt', key_ready.rt)
        thisExp.addData('key_ready.duration', key_ready.duration)
    thisExp.nextEntry()
    # the Routine "ready" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_trials = data.TrialHandler(nReps=n_trials, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_trials')
    thisExp.addLoop(loop_trials)  # add the loop to the experiment
    thisLoop_trial = loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_trial.rgb)
    if thisLoop_trial != None:
        for paramName in thisLoop_trial:
            globals()[paramName] = thisLoop_trial[paramName]
    
    for thisLoop_trial in loop_trials:
        currentLoop = loop_trials
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
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_trial.rgb)
        if thisLoop_trial != None:
            for paramName in thisLoop_trial:
                globals()[paramName] = thisLoop_trial[paramName]
        
        # --- Prepare to start Routine "trial_prep" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_prep.started', globalClock.getTime())
        # Run 'Begin Routine' code from stimuli_function
        trial = loop_trials.thisN
        
        stimuli = StimuliFunction(
            minimal_desired_duration=minimal_desired_duration,
            frequencies=frequencies,
            amplitudes=amplitudes,
            sample_rate=sample_rate,
            random_periods=random_periods,
            seed=seeds[trial]
        ).add_baseline_temp(
            baseline_temp=baseline_temp
        )#.add_plateaus(
        #    plateau_duration=plateau_duration, 
        #    n_plateaus=n_plateaus, 
        #    add_at_start=add_at_start, 
        #    add_at_end=add_at_end)
        
        # Run 'Begin Routine' code from thermoino_prep
        luigi.flush_ctc()
        luigi.init_ctc(bin_size_ms=bin_size_ms)
        luigi.create_ctc(
            temp_course=stimuli.wave,
            sample_rate=stimuli.sample_rate,
            rate_of_rise_option="mms_program")
        luigi.load_ctc()
        luigi.trigger()
        
        prep_duration = luigi.prep_ctc()[1]
        # Run 'Begin Routine' code from mouse_prep
        vas_pos_y = pixel_pos_y(
            component_pos = vas_cont.pos,
            win_size = win.size, 
            win_pos = win.pos)
            
        mouse_action.hold()
        # Run 'Begin Routine' code from rating_prep
        stimuli_clock.reset()
        vas_cont_prep.reset()
        # keep track of which components have finished
        trial_prepComponents = [vas_cont_prep]
        for thisComponent in trial_prepComponents:
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
        
        # --- Run Routine "trial_prep" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from mouse_prep
            mouse_action.check(vas_pos_y)
            # Run 'Each Frame' code from rating_prep
            # Cheap workaround to get the last rating of the prep slider
            if prep_duration - 1 < stimuli_clock.getTime():
                prep_vas_rating = vas_cont_prep.getMarkerPos()
            
            # *vas_cont_prep* updates
            
            # if vas_cont_prep is starting this frame...
            if vas_cont_prep.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                vas_cont_prep.frameNStart = frameN  # exact frame index
                vas_cont_prep.tStart = t  # local t and not account for scr refresh
                vas_cont_prep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(vas_cont_prep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vas_cont_prep.started')
                # update status
                vas_cont_prep.status = STARTED
                vas_cont_prep.setAutoDraw(True)
            
            # if vas_cont_prep is active this frame...
            if vas_cont_prep.status == STARTED:
                # update params
                pass
            
            # if vas_cont_prep is stopping this frame...
            if vas_cont_prep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > vas_cont_prep.tStartRefresh + prep_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    vas_cont_prep.tStop = t  # not accounting for scr refresh
                    vas_cont_prep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'vas_cont_prep.stopped')
                    # update status
                    vas_cont_prep.status = FINISHED
                    vas_cont_prep.setAutoDraw(False)
            
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
            for thisComponent in trial_prepComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_prep" ---
        for thisComponent in trial_prepComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_prep.stopped', globalClock.getTime())
        loop_trials.addData('vas_cont_prep.response', vas_cont_prep.getRating())
        loop_trials.addData('vas_cont_prep.rt', vas_cont_prep.getRT())
        # the Routine "trial_prep" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_vas_continuous" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_vas_continuous.started', globalClock.getTime())
        # Run 'Begin Routine' code from thermoino
        # After we have reached the starting temperature for the ctc.
        luigi.exec_ctc()
        # Run 'Begin Routine' code from imotions_event
        """ Send discrete marker for stimuli beginning """
        # TODO: add imotions marker for stimuli start and end
        imotions_event.send_marker("stimuli", "Stimuli begins")
        # Start the clock
        stimuli_clock.reset()
        # Run 'Begin Routine' code from mouse
        # Everything stays the same from the prep scale
        # Run 'Begin Routine' code from rating
        # Set the starting VAS value of the slider accordingly to the prep slider
        # This has to be done after the slider initialization
        vas_cont.startValue = prep_vas_rating
        vas_cont.reset()
        # keep track of which components have finished
        trial_vas_continuousComponents = [vas_cont]
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
            # Run 'Each Frame' code from imotions_event
            """ Stream data for pain rating """
            imotions_event.send_ratings(
                vas_cont.getMarkerPos())
            
            idx_stimuli = int(stimuli_clock.getTime()*stimuli.sample_rate)
            if idx_stimuli < len(stimuli.wave):
                imotions_event.send_temperatures(stimuli.wave[idx_stimuli])
            # Run 'Each Frame' code from mouse
            mouse_action.check(vas_pos_y)
            
            # *vas_cont* updates
            
            # if vas_cont is starting this frame...
            if vas_cont.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                vas_cont.frameNStart = frameN  # exact frame index
                vas_cont.tStart = t  # local t and not account for scr refresh
                vas_cont.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(vas_cont, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vas_cont.started')
                # update status
                vas_cont.status = STARTED
                vas_cont.setAutoDraw(True)
            
            # if vas_cont is active this frame...
            if vas_cont.status == STARTED:
                # update params
                pass
            
            # if vas_cont is stopping this frame...
            if vas_cont.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > vas_cont.tStartRefresh + stimuli.duration-frameTolerance:
                    # keep track of stop time/frame for later
                    vas_cont.tStop = t  # not accounting for scr refresh
                    vas_cont.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'vas_cont.stopped')
                    # update status
                    vas_cont.status = FINISHED
                    vas_cont.setAutoDraw(False)
            
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
        thisExp.addData('trial_vas_continuous.stopped', globalClock.getTime())
        # Run 'End Routine' code from imotions_event
        """ Send discrete marker for stimuli ending """
        imotions_event.send_marker("stimuli", "Stimuli ends")
        
        # Run 'End Routine' code from mouse
        mouse_action.release()
        loop_trials.addData('vas_cont.response', vas_cont.getRating())
        loop_trials.addData('vas_cont.rt', vas_cont.getRT())
        # the Routine "trial_vas_continuous" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_end" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_end.started', globalClock.getTime())
        # Run 'Begin Routine' code from reset_temp
        # Sometimes the Theroino takes some time to set the temperature back to baseline
        # Here, we force this with a loop for each frame.
        success = False
        # Run 'Begin Routine' code from trial_end
        # Store the trial number for conditional text box
        this_trial = loop_trials.thisN    
        text_trial_end.setText("Dieser Block ist geschafft.\n\nNun wechseln wir die Haustelle am Arm.\nBitte melden Sie sich bei der Versuchsleitung.\n\n\n(Leertaste drücken, um fortzufahren)" if this_trial != n_trials -1 else "Das Experiment ist vorbei.\n\n\nVielen Dank für Ihre Teilnahme!")
        key_trial_end.keys = []
        key_trial_end.rt = []
        _key_trial_end_allKeys = []
        # keep track of which components have finished
        trial_endComponents = [text_trial_end, key_trial_end]
        for thisComponent in trial_endComponents:
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
        
        # --- Run Routine "trial_end" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from reset_temp
            if not success and frameN % 20 == 0:
                success = luigi.set_temp(temp_baseline)[2]
            
            # *text_trial_end* updates
            
            # if text_trial_end is starting this frame...
            if text_trial_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_trial_end.frameNStart = frameN  # exact frame index
                text_trial_end.tStart = t  # local t and not account for scr refresh
                text_trial_end.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_trial_end, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_trial_end.started')
                # update status
                text_trial_end.status = STARTED
                text_trial_end.setAutoDraw(True)
            
            # if text_trial_end is active this frame...
            if text_trial_end.status == STARTED:
                # update params
                pass
            
            # *key_trial_end* updates
            waitOnFlip = False
            
            # if key_trial_end is starting this frame...
            if key_trial_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_trial_end.frameNStart = frameN  # exact frame index
                key_trial_end.tStart = t  # local t and not account for scr refresh
                key_trial_end.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_trial_end, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_trial_end.started')
                # update status
                key_trial_end.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_trial_end.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_trial_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_trial_end.status == STARTED and not waitOnFlip:
                theseKeys = key_trial_end.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_trial_end_allKeys.extend(theseKeys)
                if len(_key_trial_end_allKeys):
                    key_trial_end.keys = _key_trial_end_allKeys[-1].name  # just the last key pressed
                    key_trial_end.rt = _key_trial_end_allKeys[-1].rt
                    key_trial_end.duration = _key_trial_end_allKeys[-1].duration
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
            for thisComponent in trial_endComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_end" ---
        for thisComponent in trial_endComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_end.stopped', globalClock.getTime())
        # check responses
        if key_trial_end.keys in ['', [], None]:  # No response was made
            key_trial_end.keys = None
        loop_trials.addData('key_trial_end.keys',key_trial_end.keys)
        if key_trial_end.keys != None:  # we had a response
            loop_trials.addData('key_trial_end.rt', key_trial_end.rt)
            loop_trials.addData('key_trial_end.duration', key_trial_end.duration)
        # the Routine "trial_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_next" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_next.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (this_trial == n_trials -1)
        key_trial_next.keys = []
        key_trial_next.rt = []
        _key_trial_next_allKeys = []
        # keep track of which components have finished
        trial_nextComponents = [key_trial_next, text_trial_next]
        for thisComponent in trial_nextComponents:
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
        
        # --- Run Routine "trial_next" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_trial_next* updates
            waitOnFlip = False
            
            # if key_trial_next is starting this frame...
            if key_trial_next.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_trial_next.frameNStart = frameN  # exact frame index
                key_trial_next.tStart = t  # local t and not account for scr refresh
                key_trial_next.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_trial_next, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_trial_next.started')
                # update status
                key_trial_next.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_trial_next.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_trial_next.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_trial_next.status == STARTED and not waitOnFlip:
                theseKeys = key_trial_next.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_trial_next_allKeys.extend(theseKeys)
                if len(_key_trial_next_allKeys):
                    key_trial_next.keys = _key_trial_next_allKeys[-1].name  # just the last key pressed
                    key_trial_next.rt = _key_trial_next_allKeys[-1].rt
                    key_trial_next.duration = _key_trial_next_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_trial_next* updates
            
            # if text_trial_next is starting this frame...
            if text_trial_next.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_trial_next.frameNStart = frameN  # exact frame index
                text_trial_next.tStart = t  # local t and not account for scr refresh
                text_trial_next.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_trial_next, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_trial_next.started')
                # update status
                text_trial_next.status = STARTED
                text_trial_next.setAutoDraw(True)
            
            # if text_trial_next is active this frame...
            if text_trial_next.status == STARTED:
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
            for thisComponent in trial_nextComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_next" ---
        for thisComponent in trial_nextComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_next.stopped', globalClock.getTime())
        # check responses
        if key_trial_next.keys in ['', [], None]:  # No response was made
            key_trial_next.keys = None
        loop_trials.addData('key_trial_next.keys',key_trial_next.keys)
        if key_trial_next.keys != None:  # we had a response
            loop_trials.addData('key_trial_next.rt', key_trial_next.rt)
            loop_trials.addData('key_trial_next.duration', key_trial_next.duration)
        # the Routine "trial_next" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed n_trials repeats of 'loop_trials'
    
    # Run 'End Experiment' code from all_variables
    # Moved to trial_end component so that it gets called later on
    # close_logger(psychopy_logger)
    # Run 'End Experiment' code from imotions_control
    imotions_control.end_study()
    imotions_control.close()
    # Run 'End Experiment' code from thermoino
    luigi.close()
    # Run 'End Experiment' code from imotions_event
    """ Close event recieving API connection """
    imotions_event.end_study()
    imotions_event.close()
    # Run 'End Experiment' code from trial_end
    close_logger(psychopy_logger)
    
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
