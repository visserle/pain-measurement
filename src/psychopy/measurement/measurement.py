#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.1),
    on November 22, 2023, at 10:46

Further modifications were made:
    - A confirmation window before window creation to confirm the start of the experiment
        -> This is needed to stop psychopy going into fullscreen mode before iMotions did the calibration (which is also in fullscreen mode)
    - Global variables:
        - win and inputs are created in the run function
        - expInfo, thisExp and logFile are created in the main function
    - Logging:
        - Import the psychopy logging module not as logging but as psychopy.logging
    - Mouse:
        - Remind psychopy to hide the mouse in the run function

Note: These are minimal modifications to the auto-generated experiment, so there might be a better way to do this.
"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
import psychopy
from psychopy import gui, visual, core, data, event, clock
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np
import os
import sys
import platform

import psychopy.iohub as io
from psychopy.hardware import keyboard

from src.psychopy.psychopy_utils import ask_for_confirmation, rgb255_to_rgb_psychopy, runs_psychopy_path

import json

# Logging
import logging
from src.log_config import configure_logging, close_root_logging


# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))

configure_logging(file_path=runs_psychopy_path(_thisDir, "logs"), ignore_libs=['PIL'])

config_path = os.path.join(_thisDir, 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)


# Store info about the experiment session
psychopyVersion = '2023.2.1'
expName = 'measurement'  # from the Builder filename that created this script
expInfo = {
    'date': data.getDateStr(), 
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Run 'Before Experiment' code from all_variables

# Colors
element_color = rgb255_to_rgb_psychopy(config['psychopy']['element_color'])
marker_color = rgb255_to_rgb_psychopy(config['psychopy']['marker_color'])

# Get all variables from config.json
expInfo['expName'] = config['psychopy']['name']
expInfo["dummy"] = config['psychopy']['dummy']

# Import dummy scripts for debugging if specified
if expInfo["dummy"] is True:
    from src.psychopy.imotions_dummy import RemoteControliMotions, EventRecievingiMotions
    from src.psychopy.thermoino_dummy import ThermoinoComplexTimeCourses
    expInfo.update(config['psychopy']['dummy_participant'])
    participant_info = {}
    participant_info.update(config['psychopy']['dummy_participant'])
    
elif expInfo["dummy"] is False:
    from src.psychopy.imotions import RemoteControliMotions, EventRecievingiMotions
    from src.psychopy.thermoino import ThermoinoComplexTimeCourses
    from src.psychopy.participant_data import read_last_participant
    participant_info = read_last_participant()
    expInfo.update(participant_info)
        
port = config['thermoino']['port']
mms_baseline = config['thermoino']['mms_baseline'] # has to be the same as in MMS (not the same as baseline_temp)
mms_rate_of_rise = config['thermoino']['mms_rate_of_rise'] # has to be the same as in MMS
bin_size_ms = config['thermoino']['bin_size_ms']

# iMotions # TODO: set start_study_mode to NoPrompt?
start_study_mode = config['imotions']['start_study_mode']

# Stimuli
seeds = config['stimuli']['seeds']
n_trials = len(seeds)
this_trial = None # only for initializing
minimal_desired_duration = config['stimuli']['minimal_desired_duration'] # in seconds
periods = config['stimuli']['periods'] # [0] is the baseline and [1] the modulation; in seconds
frequencies = 1./np.array(periods)
sample_rate = config['stimuli']['sample_rate']
desired_big_decreases = config['stimuli']['desired_big_decreases']
random_periods = config['stimuli']['random_periods']
plateau_duration = config['stimuli']['plateau_duration']
n_plateaus = config['stimuli']['n_plateaus']

baseline_temp = participant_info['baseline_temp'] # @ VAS 35
temp_range = participant_info['temp_range'] # VAS 70 - VAS 0


# Time
stimuli_clock = core.Clock()
iti_duration = 8  + np.random.randint(0, 5)
vas_labels = ("Keine\nSchmerzen", "Sehr starke Schmerzen")
# Run 'Before Experiment' code from mouse_instruction
import src.psychopy.mouse_action as mouse_action
from src.psychopy.mouse_action import pixel_pos_y
# Run 'Before Experiment' code from stimuli_function
from src.psychopy.stimuli_function import StimuliFunction
# Run 'Before Experiment' code from imotions_event
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
        originPath='G:\\Meine Ablage\\PhD\\Code\\pain-measurement\\experiments\\measurement\\measurement.py',
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
    logFile = psychopy.logging.LogFile(filename+'.log', level=psychopy.logging.EXP)
    psychopy.logging.console.setLevel(psychopy.logging.WARNING)  # this outputs to the screen, not a file
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
            size=[1920, 1200], fullscr=True, screen=0,
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
    win.mouseVisible = False
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


def run(expInfo, thisExp, globalClock=None, thisSession=None):
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

    # Run 'Begin Experiment' code from imotions_control
    imotions_control = RemoteControliMotions(
        study = expInfo['expName'],
        participant = expInfo['participant'],
        age = expInfo['age'],
        gender = expInfo['gender']
    )
    
    imotions_control.connect()
    imotions_control.start_study(mode=start_study_mode)

    if platform.system() != 'Darwin':
        response = ask_for_confirmation()
        if not response:
            quit(thisExp)

    # Window creation
    global win
    global inputs

    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    # get device handles from dict of input devices

    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']

    # mark experiment as started
    thisExp.status = STARTED


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
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Willkommen zum Haupt-Experiment!\n\n\nBitte drücken Sie die Leertaste.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_2" ---
    text_welcome_2 = visual.TextStim(win=win, name='text_welcome_2',
        text='Mit diesem Experiment möchten wir mit Hilfe Ihrer Daten Schmerzen objektiv messbar machen.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "welcome_3" ---
    text_welcome_3 = visual.TextStim(win=win, name='text_welcome_3',
        text='Dazu erhalten Sie im Folgenden Schmerzen mit schwankender Intensität über insgesamt 12 Blöcke á 4 Minuten.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instruction" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Ihre Aufgabe ist es, Ihren Schmerz durchgehend zu bewerten.\n\nDies geschieht über eine Skala, die Sie auf der folgenden Seite sehen.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruction = keyboard.Keyboard()

    # --- Initialize components for Routine "instruction_1" ---
    text_instruction_1 = visual.TextStim(win=win, name='text_instruction_1',
        text='\n\n\n\n\n\n\n\n\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruction_1 = keyboard.Keyboard()
    vas_cont_instruction_1 = visual.Slider(win=win, name='vas_cont_instruction_1',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor=element_color, markerColor=marker_color, lineColor=element_color, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "instruction_2" ---
    text_instruction_2 = visual.TextStim(win=win, name='text_instruction_2',
        text='Bitte bewegen Sie Ihren Mauszeiger auf der Skala hin und her!\n\n\n\n\n\n\n\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruction_2 = keyboard.Keyboard()
    vas_cont_instruction_2 = visual.Slider(win=win, name='vas_cont_instruction_2',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor=element_color, markerColor=marker_color, lineColor=element_color, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "instruction_3" ---
    text_instruction_3 = visual.TextStim(win=win, name='text_instruction_3',
        text='Die Beschriftungen "Keine Schmerzen" und "Sehr starke Schmerzen" entsprechen dabei Ihren Werten aus der Kalibrierung. Nutzen Sie die gesamte Breite der Skala!\n\n\n\n\n\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_instruction_3 = keyboard.Keyboard()
    vas_cont_instruction_3 = visual.Slider(win=win, name='vas_cont_instruction_3',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor=element_color, markerColor=marker_color, lineColor=element_color, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "ready" ---
    text_ready = visual.TextStim(win=win, name='text_ready',
        text='Als Nächstes startet das Experiment.\n\nSollten Sie noch Fragen haben, richten Sie sich bitte jetzt an die Versuchsleitung.\n\n\n(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_ready = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_prep" ---
    vas_cont_prep = visual.Slider(win=win, name='vas_cont_prep',
        startValue=50, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor=element_color, markerColor=marker_color, lineColor=element_color, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # --- Initialize components for Routine "trial_vas_continuous" ---
    # Run 'Begin Experiment' code from thermoino
    luigi = ThermoinoComplexTimeCourses(
        port=port, 
        mms_baseline=mms_baseline, 
        mms_rate_of_rise=mms_rate_of_rise)
    
    luigi.connect()
    # Run 'Begin Experiment' code from imotions_event
    """ Connect with event recieving API """
    imotions_event = EventRecievingiMotions()
    imotions_event.connect()
    imotions_event.start_study()
    
    # create a clock
    stimuli_clock = core.Clock()
    vas_cont = visual.Slider(win=win, name='vas_cont',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=vas_labels, ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor=element_color, markerColor=marker_color, lineColor=element_color, colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # --- Initialize components for Routine "trial_end" ---
    # Run 'Begin Experiment' code from trial_end
    this_trial = None
    text_trial_end = visual.TextStim(win=win, name='text_trial_end',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_trial_end = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_next" ---
    key_trial_next = keyboard.Keyboard()
    text_trial_next = visual.TextStim(win=win, name='text_trial_next',
        text='(Leertaste drücken, um fortzufahren)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=element_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    psychopy.logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    win.mouseVisible = False # "remind" psychopy to hide the mouse
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
    key_instruction.keys = []
    key_instruction.rt = []
    _key_instruction_allKeys = []
    # keep track of which components have finished
    instructionComponents = [text_instruction, key_instruction]
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
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruction_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_1.started', globalClock.getTime())
    # Run 'Begin Routine' code from mouse_instruction
    vas_pos_y = pixel_pos_y(
        component_pos = vas_cont.pos,
        win_size = win.size, 
        win_pos = win.pos)
        
    mouse_action.hold()
    key_instruction_1.keys = []
    key_instruction_1.rt = []
    _key_instruction_1_allKeys = []
    vas_cont_instruction_1.reset()
    # keep track of which components have finished
    instruction_1Components = [text_instruction_1, key_instruction_1, vas_cont_instruction_1]
    for thisComponent in instruction_1Components:
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

    # --- Run Routine "instruction_1" ---
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
        
        # *text_instruction_1* updates
        
        # if text_instruction_1 is starting this frame...
        if text_instruction_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction_1.frameNStart = frameN  # exact frame index
            text_instruction_1.tStart = t  # local t and not account for scr refresh
            text_instruction_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction_1.started')
            # update status
            text_instruction_1.status = STARTED
            text_instruction_1.setAutoDraw(True)
        
        # if text_instruction_1 is active this frame...
        if text_instruction_1.status == STARTED:
            # update params
            pass
        
        # *key_instruction_1* updates
        waitOnFlip = False
        
        # if key_instruction_1 is starting this frame...
        if key_instruction_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruction_1.frameNStart = frameN  # exact frame index
            key_instruction_1.tStart = t  # local t and not account for scr refresh
            key_instruction_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruction_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruction_1.started')
            # update status
            key_instruction_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruction_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruction_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruction_1.status == STARTED and not waitOnFlip:
            theseKeys = key_instruction_1.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruction_1_allKeys.extend(theseKeys)
            if len(_key_instruction_1_allKeys):
                key_instruction_1.keys = _key_instruction_1_allKeys[-1].name  # just the last key pressed
                key_instruction_1.rt = _key_instruction_1_allKeys[-1].rt
                key_instruction_1.duration = _key_instruction_1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *vas_cont_instruction_1* updates
        
        # if vas_cont_instruction_1 is starting this frame...
        if vas_cont_instruction_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            vas_cont_instruction_1.frameNStart = frameN  # exact frame index
            vas_cont_instruction_1.tStart = t  # local t and not account for scr refresh
            vas_cont_instruction_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vas_cont_instruction_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vas_cont_instruction_1.started')
            # update status
            vas_cont_instruction_1.status = STARTED
            vas_cont_instruction_1.setAutoDraw(True)
        
        # if vas_cont_instruction_1 is active this frame...
        if vas_cont_instruction_1.status == STARTED:
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
        for thisComponent in instruction_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "instruction_1" ---
    for thisComponent in instruction_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction_1.stopped', globalClock.getTime())
    # check responses
    if key_instruction_1.keys in ['', [], None]:  # No response was made
        key_instruction_1.keys = None
    thisExp.addData('key_instruction_1.keys',key_instruction_1.keys)
    if key_instruction_1.keys != None:  # we had a response
        thisExp.addData('key_instruction_1.rt', key_instruction_1.rt)
        thisExp.addData('key_instruction_1.duration', key_instruction_1.duration)
    thisExp.nextEntry()
    thisExp.addData('vas_cont_instruction_1.response', vas_cont_instruction_1.getRating())
    thisExp.addData('vas_cont_instruction_1.rt', vas_cont_instruction_1.getRT())
    # the Routine "instruction_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()

    # --- Prepare to start Routine "instruction_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_3.started', globalClock.getTime())
    key_instruction_3.keys = []
    key_instruction_3.rt = []
    _key_instruction_3_allKeys = []
    vas_cont_instruction_3.reset()
    # keep track of which components have finished
    instruction_3Components = [text_instruction_3, key_instruction_3, vas_cont_instruction_3]
    for thisComponent in instruction_3Components:
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
    
    # --- Prepare to start Routine "instruction_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_2.started', globalClock.getTime())
    # Run 'Begin Routine' code from mouse_instruction
    vas_pos_y = pixel_pos_y(
        component_pos = vas_cont.pos,
        win_size = win.size, 
        win_pos = win.pos)
        
    mouse_action.hold()
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
        # Run 'Each Frame' code from mouse_instruction
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
    
    # --- Prepare to start Routine "instruction_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction_3.started', globalClock.getTime())
    key_instruction_3.keys = []
    key_instruction_3.rt = []
    _key_instruction_3_allKeys = []
    vas_cont_instruction_3.reset()
    # keep track of which components have finished
    instruction_3Components = [text_instruction_3, key_instruction_3, vas_cont_instruction_3]
    for thisComponent in instruction_3Components:
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
    
    # --- Run Routine "instruction_3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from mouse_instruction_3
        mouse_action.check(vas_pos_y)
        
        # *text_instruction_3* updates
        
        # if text_instruction_3 is starting this frame...
        if text_instruction_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction_3.frameNStart = frameN  # exact frame index
            text_instruction_3.tStart = t  # local t and not account for scr refresh
            text_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction_3.started')
            # update status
            text_instruction_3.status = STARTED
            text_instruction_3.setAutoDraw(True)
        
        # if text_instruction_3 is active this frame...
        if text_instruction_3.status == STARTED:
            # update params
            pass
        
        # *key_instruction_3* updates
        waitOnFlip = False
        
        # if key_instruction_3 is starting this frame...
        if key_instruction_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruction_3.frameNStart = frameN  # exact frame index
            key_instruction_3.tStart = t  # local t and not account for scr refresh
            key_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruction_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruction_3.started')
            # update status
            key_instruction_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruction_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruction_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruction_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instruction_3.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruction_3_allKeys.extend(theseKeys)
            if len(_key_instruction_3_allKeys):
                key_instruction_3.keys = _key_instruction_3_allKeys[-1].name  # just the last key pressed
                key_instruction_3.rt = _key_instruction_3_allKeys[-1].rt
                key_instruction_3.duration = _key_instruction_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *vas_cont_instruction_3* updates
        
        # if vas_cont_instruction_3 is starting this frame...
        if vas_cont_instruction_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            vas_cont_instruction_3.frameNStart = frameN  # exact frame index
            vas_cont_instruction_3.tStart = t  # local t and not account for scr refresh
            vas_cont_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vas_cont_instruction_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vas_cont_instruction_3.started')
            # update status
            vas_cont_instruction_3.status = STARTED
            vas_cont_instruction_3.setAutoDraw(True)
        
        # if vas_cont_instruction_3 is active this frame...
        if vas_cont_instruction_3.status == STARTED:
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
        for thisComponent in instruction_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction_3" ---
    for thisComponent in instruction_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction_3.stopped', globalClock.getTime())
    # Run 'End Routine' code from mouse_instruction_3
    mouse_action.release()
    # check responses
    if key_instruction_3.keys in ['', [], None]:  # No response was made
        key_instruction_3.keys = None
    thisExp.addData('key_instruction_3.keys',key_instruction_3.keys)
    if key_instruction_3.keys != None:  # we had a response
        thisExp.addData('key_instruction_3.rt', key_instruction_3.rt)
        thisExp.addData('key_instruction_3.duration', key_instruction_3.duration)
    thisExp.nextEntry()
    thisExp.addData('vas_cont_instruction_3.response', vas_cont_instruction_3.getRating())
    thisExp.addData('vas_cont_instruction_3.rt', vas_cont_instruction_3.getRT())
    # the Routine "instruction_3" was not non-slip safe, so reset the non-slip timer
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
        
        seed = seeds[loop_trials.thisN]
        # TODO: might needs a change once the trials randomization is implemented
        logging.info(f"Psychopy starts trial ({loop_trials.thisN + 1}/{n_trials} with seed {seed}") 
        
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
            temp_range=temp_range,
            sample_rate=sample_rate,
            desired_big_decreases=desired_big_decreases,
            random_periods=random_periods,
            seed=seed
        ).add_baseline_temp(
            baseline_temp=baseline_temp
        ).add_plateaus(
            plateau_duration=plateau_duration, 
            n_plateaus=n_plateaus
        ).generalize_big_decreases()
        
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
        imotions_event.send_stimulus_markers(seed)
        
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
                round(vas_cont.getMarkerPos(),3))
            
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
        imotions_event.send_stimulus_markers(seed)
        
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
        # Sometimes the Thermoino takes some time to set the temperature back to baseline
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
            if not success and frameN % 10 == 0:
                success = luigi.set_temp(mms_baseline)[2]
            
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
    # Moved close_root_logging() to trial_end component
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
    close_root_logging()
    
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
    psychopy.logging.flush()


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
    psychopy.logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


def main():
    global expInfo
    global thisExp
    global logFile

    # expInfo = showExpInfoDlg(expInfo=expInfo) # we do not ask for subject info -> its all in the config file
    thisExp = setupData(expInfo=expInfo,dataDir=str(runs_psychopy_path(_thisDir, "data")))

    logFile = setupLogging(filename=thisExp.dataFileName)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)


# if running this experiment as a script...
if __name__ == '__main__':
    main()
