2032.08.31
- ramp off time should not be part of the stimilus length
-> calculate time based on rate_of_rise and add it in every trial
- find out how to manage MMS max time, maybe keep calibration in small window?


Notes
- add utils to the path of psychopy

Define by hand:
- port of Thermoino


default frame rate is 60,
but in the experiment the frame rate will often change to 30
-> that's why nothing is dependent on frame counts, but on clocks / time


___
mpad1:

TODO
- find seeds
- define baseline_temp variable in accordance to calibration
- remove debugging restrictions of stimuli
- scrutinize loggings in console + files
- use trial_prep, also for finding the starting value of the vas_cont
- try it on yourself

- edit resulting .py by hand

- perhaps change the path to src/experiments/psychopy.
-> and loggings to runs/psychopy?