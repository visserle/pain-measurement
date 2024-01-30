from expyriment import design, control, stimuli, misc
control.set_develop_mode(True)

exp = design.Experiment(name="Stimuli Demo")

control.initialize(exp)

control.start()

target = stimuli.TextLine(text="I am a text!", text_size=80)
stimuli.FixCross().present()
target.preload()
exp.clock.wait(1000)
target.present()
exp.clock.wait(1000)

control.end()