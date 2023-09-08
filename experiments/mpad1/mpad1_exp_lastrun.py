﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.1),
    on August 24, 2023, at 12:53
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
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'pain_threshold': '38',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Run 'Before Experiment' code from stimuli_function
from src.experiments.stimuli_function import StimuliFunction
# Run 'Before Experiment' code from imotions_control
from src.experiments.imotions import RemoteControliMotions
# Run 'Before Experiment' code from thermoino
from src.experiments.thermoino import ThermoinoComplexTimeCourses
# Run 'Before Experiment' code from mouse
import src.experiments.mouse_action as mouse_action
from src.experiments.mouse_action import pixel_pos_y
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
        originPath='G:\\Meine Ablage\\PhD\\Code\\mpad-pilot\\experiments\\mpad1\\mpad1_exp_lastrun.py',
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
    
    # --- Initialize components for Routine "welcome_screen" ---
    # Run 'Begin Experiment' code from stimuli_function
    minimal_desired_duration = 10 # in seconds
    periods = [67, 20] # [0] is the baseline and [1] the modulation; in seconds
    frequencies = 1./np.array(periods)
    amplitudes = [1, 1.5] # temp range is 2 * sum(amplitudes): max @ VAS 70, min @ VAS 0
    sample_rate = 60
    seed = 463 # use None for random seed
    baseline_temp = 39.2 # @ VAS 35
    
    stimuli = StimuliFunction(
        minimal_desired_duration,
        frequencies,
        amplitudes,
        sample_rate,
        random_periods=False, # for debugging
        seed=seed
    ).add_baseline_temp(
        baseline_temp
    )#.add_plateaus(
    #    plateau_duration=20, 
    #    n_plateaus=4, 
    #    add_at_start="random", 
    #    add_at_end=True)
    # Run 'Begin Experiment' code from imotions_control
    imotions_control = RemoteControliMotions(
        study = expName,
        participant = expInfo['participant'],
    )
    
    imotions_control.connect()
    imotions_control.start_study()
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Willkommen zum Experiment!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_start_exp = visual.TextStim(win=win, name='text_start_exp',
        text='Das Experiment beginnt mit Tastendruck.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_welcome = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial_vas_continuous" ---
    # Run 'Begin Experiment' code from thermoino
    port = "COM13" # top usb port on the front, use list_com_ports() to find out
    luigi = ThermoinoComplexTimeCourses(
        port=port, 
        temp_baseline=32, 
        rate_of_rise=5)
    
    luigi.connect()
    luigi.init_ctc(bin_size_ms=500)
    luigi.create_ctc(
        temp_course=stimuli.wave,
        sample_rate=stimuli.sample_rate,
        rate_of_rise_option="mms_program")
    luigi.load_ctc()
    luigi.trigger()
    luigi.prep_ctc() # TODO: start at VAS 35, not optimal
    vas_cont = visual.Slider(win=win, name='vas_cont',
        startValue=50, size=(1.0, 0.1), pos=(0, -0.1), units=win.units,
        labels=("Kein Schmerz","Stärkste vorstellbare Schmerzen"), ticks=(0, 100), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    text_vas_cont = visual.TextStim(win=win, name='text_vas_cont',
        text='',
        font='Open Sans',
        pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from imotions_event
    """ Connect with event recieving API """
    imotions_event = EventRecievingiMotions()
    imotions_event.connect()
    imotions_event.start_study
    
    # create a clock
    stimuli_clock = core.Clock()
    
    # --- Initialize components for Routine "blank500" ---
    
    # --- Initialize components for Routine "goodbye_screen" ---
    text_goodbye = visual.TextStim(win=win, name='text_goodbye',
        text='Vielen Dank für Ihre Versuchsteilnahme!',
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
    
    # --- Prepare to start Routine "welcome_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome_screen.started', globalClock.getTime())
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
    thisExp.addData('welcome_screen.stopped', globalClock.getTime())
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
    thisExp.addData('trial_vas_continuous.started', globalClock.getTime())
    # Run 'Begin Routine' code from thermoino
    luigi.exec_ctc()
    vas_cont.reset()
    # Run 'Begin Routine' code from mouse
    vas_pos_y = pixel_pos_y(
        component_pos = vas_cont.pos,
        win_size = win.size, 
        win_pos = win.pos)
        
    mouse_action.hold()
    # Run 'Begin Routine' code from imotions_event
    """ Send discrete marker for stimuli beginning """
    imotions_event.send_marker("stimuli", "Stimuli begins")
    # Start the clock
    stimuli_clock.reset()
    # keep track of which components have finished
    trial_vas_continuousComponents = [vas_cont, text_vas_cont]
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
        # Run 'Each Frame' code from mouse
        mouse_action.check(vas_pos_y*1.1) # 1.1 to exactly hit the slider
        
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
            text_vas_cont.setText(f"Dies ist eine kontinuierliche VAS.\n\nIhr momentaner Schmerz liegt bei: {vas_cont.getMarkerPos():.0f}" if vas_cont.getMarkerPos() is not None else "Dies ist eine kontinuierliche VAS.\n\nIhr momentaner  Schmerz liegt bei: --", log=False)
        
        # if text_vas_cont is stopping this frame...
        if text_vas_cont.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_vas_cont.tStartRefresh + stimuli.duration-frameTolerance:
                # keep track of stop time/frame for later
                text_vas_cont.tStop = t  # not accounting for scr refresh
                text_vas_cont.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_vas_cont.stopped')
                # update status
                text_vas_cont.status = FINISHED
                text_vas_cont.setAutoDraw(False)
        # Run 'Each Frame' code from imotions_event
        """ Stream data for pain rating """
        imotions_event.send_ratings(
            vas_cont.getMarkerPos())
        
        idx_stimuli = int(stimuli_clock.getTime()*stimuli.sample_rate)
        if idx_stimuli < len(stimuli.wave):
            imotions_event.send_temperatures(stimuli.wave[idx_stimuli])
        
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
    # Run 'End Routine' code from thermoino
    luigi.set_temp(32)
    luigi.close()
    thisExp.addData('vas_cont.response', vas_cont.getRating())
    thisExp.addData('vas_cont.rt', vas_cont.getRT())
    # Run 'End Routine' code from mouse
    mouse_action.release()
    # Run 'End Routine' code from imotions_event
    """ Send discrete marker for stimuli ending """
    imotions_event.send_marker("stimuli", "Stimuli ends")
    
    # the Routine "trial_vas_continuous" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank500" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('blank500.started', globalClock.getTime())
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
    thisExp.addData('blank500.stopped', globalClock.getTime())
    # the Routine "blank500" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "goodbye_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('goodbye_screen.started', globalClock.getTime())
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
    thisExp.addData('goodbye_screen.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    # Run 'End Experiment' code from imotions_control
    imotions_control.end_study()
    imotions_control.close()
    # Run 'End Experiment' code from imotions_event
    """ Close event recieving API connection """
    imotions_event.end_study()
    imotions_event.close()
    
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