*work in progress*

TODO: 
- path to the main experiment script
- calibration description
- rewrite flow

# Modules
*Short description on how the experimental modules interact with each other.*

Modules:
1. stimuli_function.py
1. thermoino.py
1. imotions.py
1. mouse_action.py
1. calibration.py
1. logger.py

-> all modules are called by the main psychopy experiment script under the folder `mpad-pilot/experiments/`


## stimuli_function.py
This module contains the class that generates the stimuli (temperature course) for the experiment. 
The function returns a stimuli object that is used by the thermoino module to control the temperature.

## thermoino.py
This module contains the class that controls the thermoino device. 
It sends commands to the Thermoino which is in contact with the Thermode.
It has to be set up in accordance with the Medoc Main Station (MMS) program.
The thermoino class receives informations about the stimuli. 
The thermoino class then controls the temperature according to the stimuli object.

## imotions.py
This module contains two classes: one for remote control of iMotions and one for event / marker sending.
The remote control class is used to start and stop the recording of iMotions.
The event / marker class is used to send 1. events (continuous) for the temperature and rating curves to iMotions and 2. markers (discrete) which can be useful for later analysis.

## mouse_action.py
This module contains the class that controls the mouse for continuous ratings in the experiment.
It is a simple a workaround for using the standard psychopy slider component without the need to press any buttons. 
The slider solely relies on the x-coordinate of the mouse as input for the rating while the mouse is being clicked and held automatically.

## calibration.py
This module contains the class that controls the calibration of the temperatures for the heat pain experiment. It is used in calibration psychopy experiment.

(?) The calibration is done in the beginning of the experiment and is saved in a seperate file. The calibration file is then used by the stimuli function to generate the stimuli for the experiment.

## logger.py
The logger module contains the class that logs the data from the experiment. It is used in the main psychopy experiment script. The logger class saves the data in a seperate logfile and outputs to the console of the psychopy runner. 
