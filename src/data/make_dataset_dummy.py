# work in progress

# TODO:
# - add extra functionalities from make_dataset.py
# - make reshaping more elegant, e.g. data = np.reshape(data, (self.n_samples, -1, data.shape[-1]))

from pathlib import Path
import numpy as np
from numpy.random import randint
import pandas as pd
import neurokit2 as nk
import random
from matplotlib import pyplot as plt

class SimulateTrial:
    '''
    This class generates physiological dummy signals for a single dummy trial.

    A single dummy trial consists of two phases: a "ramp up" of temperature and a "ramp off". 
    These are also the labels of the target array, where 0 corresponds to the "ramp up" phase and 1 to the "ramp off" phase.
    Note that there are no actual ramps in the data and the difference between the two ramps is only in the constants of the dummy physiological profiles.
    This is only dummy data for testing purposes.
    
    Attributes
    ----------
    duration: int, optional
        Duration of the trial in seconds (default is 10)
    sampling_rate: int, optional
        Sampling rate of the physiological signals (default is 50)
    ramp_up: dict, optional
        Physiological profile at the start of the trial. It includes heart rate, heart rate standard deviation, respiratory rate, and drift.
    ramp_off: dict, optional
        Physiological profile at the end of the trial. It includes heart rate, heart rate standard deviation, respiratory rate, and drift.
    '''
    duration = 10  # s
    sampling_rate = 50

    def __init__(self,
            # default physiological profile for 1 subject
            ramp_up={
                "heart_rate": 70,
                "heart_rate_std": 2,
                "respiratory_rate": 15,
                "drift": 0.1
            },
            ramp_off={
                "heart_rate": 65,
                "heart_rate_std": 5,
                "respiratory_rate": 12,
                "drift": -0.1
            }):
        self.ramp_up = ramp_up
        self.ramp_off = ramp_off

    def signals(self, sampling_rate=sampling_rate, duration=duration):
        '''
        This method simulates physiological signals for ECG, RSP and SCL using the neurokit2 library.

        Parameters
        ----------
        sampling_rate: int, optional
            Sampling rate of the physiological signals (default is class attribute sampling_rate)
        duration: int, optional
            Duration of the trial in seconds (default is class attribute duration)

        Returns
        -------
        numpy.ndarray
            3D array containing the physiological signals. The shape is [trials (up, off), time steps, features]
        '''
        ecg = [
            nk.ecg_simulate(
                duration=duration, sampling_rate=sampling_rate, noise=0.1,
                method="daubechies",
                heart_rate=self.ramp_up["heart_rate"],
                heart_rate_std=self.ramp_up["heart_rate_std"]),
            nk.ecg_simulate(
                duration=duration, sampling_rate=sampling_rate, noise=0.1,
                method="daubechies",
                heart_rate=self.ramp_off["heart_rate"],
                heart_rate_std=self.ramp_off["heart_rate_std"])
        ]
        rsp = [
            nk.rsp_simulate(
                duration=duration, sampling_rate=sampling_rate,
                respiratory_rate=self.ramp_up["respiratory_rate"]),
            nk.rsp_simulate(
                duration=duration, sampling_rate=sampling_rate,
                respiratory_rate=self.ramp_off["respiratory_rate"])
        ]
        scl = [
            nk.eda_process(
                nk.eda_simulate(
                    duration=duration, sampling_rate=sampling_rate, noise=0.1,
                    scr_number=2, drift=self.ramp_up["drift"]),
                sampling_rate=sampling_rate)[0]["EDA_Tonic"],
            nk.eda_process(
                nk.eda_simulate(
                    duration=duration, sampling_rate=sampling_rate, noise=0.1,
                    scr_number=2, drift=self.ramp_off["drift"]),
                sampling_rate=sampling_rate)[0]["EDA_Tonic"]
        ]
        # output shape is [trials (up,off), time steps, features]
        data = np.stack([ecg, rsp, scl], axis=2)

        return data


class SimulateSubject:
    '''
    This class generates an artificial physiological profile for a single subject.

    Attributes
    ----------
    seed: int
        Seed for the random number generator. One seed corresponds to one subject.
    ramp_up: dict
        Physiological profile for the first half of a trial.
    ramp_off: dict
        Physiological profile for the second half of a trial.
    '''
    def __init__(self, seed):
        # 1 seed for 1 subject
        self.seed = seed
        self.ramp_up, self.ramp_off = self.physio()

    def physio(self):
        '''
        This method generates a physiological profile for a subject.

        Returns
        -------
        tuple
            Two dictionaries representing the physiological profiles for the ramp_up and the ramp_off phases.
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)

        ramp_up = {
            "heart_rate": random.choice(range(68, 71)),
            "heart_rate_std": random.choice(range(2, 7)),
            "respiratory_rate": random.choice(range(12, 15)),
            "drift": random.choice(range(0, 2))/10
        }
        ramp_off = {
            "heart_rate": random.choice(range(66, 69)),
            "heart_rate_std": random.choice(range(4, 9)),
            "respiratory_rate": random.choice(range(12, 15)),
            "drift": -random.choice(range(0, 2))/10
        }

        return ramp_up, ramp_off


class SimulateExperiment(SimulateSubject, SimulateTrial):
    '''
    This class generates experimental data over several trials using distinct physiological profiles.

    It inherits from the SimulateSubject and SimulateTrial classes to simulate a whole experiment with several, distinct subjects and noumerous trials per subject.

    Attributes
    ----------
    n_subjects: int, optional
        Number of subjects in the experiment (default is 4)
    n_trials: int, optional
        Number of trials per subject (default is 3). One trial consists of two phases: a "ramp up" of temperature and a "ramp off".
    n_subject_samples: int
        Number of samples per subject (ramp up & off). This is equal to n_trials*2.
    n_samples: int
        Number of samples in total
    data: numpy.ndarray
        3D array containing the experimental data. The shape is [samples, time-steps, features]
    target: numpy.ndarray
        1D array containing the target labels for the experimental data.
    groups: numpy.ndarray
        1D array containing the group labels for the experimental data. Each subject is assigned a unique group label.
    '''
    def __init__(
            self,
            n_subjects=4,
            n_trials=3):

        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.n_subject_samples = n_trials*2  # ramp up & off
        self.n_samples = n_subjects*self.n_subject_samples

        self.data = self.simulate_data()
        self.target = self.simulate_target()
        self.groups = np.repeat(np.arange(0, n_subjects), n_trials*2)

    def simulate_data(self):
        '''
        This method generates experimental data for all subjects and trials.

        Returns
        -------
        numpy.ndarray
            3D array containing the experimental data. The shape is [samples, time-steps, features]
        '''
        data = []

        for i in range(self.n_subjects):
            ramp_up = SimulateSubject(i).ramp_up
            ramp_off = SimulateSubject(i).ramp_off
            data_subject = []
            for j in range(self.n_trials):
                temp = SimulateTrial(
                    ramp_up=ramp_up,
                    ramp_off=ramp_off).signals()
                data_subject.append(temp)

            data.append(data_subject)

        # in the following we reshape simulated data into a 3D array
        data = np.array(data)
        # reshape without extra dimension for ramp up & off
        data = np.reshape(data, (
            self.n_subjects,
            self.n_subject_samples,
            data.shape[-2],
            data.shape[-1]
        ))
        # reshape without extra dimension for subject & subject_sample
        # returns [samples, time-steps, features]
        data = np.reshape(data, (
            self.n_samples,
            data.shape[-2],
            data.shape[-1]
        ))

        return data

    def simulate_target(self):
        '''
        This method generates the target array for the experimental data, 
        where 0 corresponds to the "ramp up" phase and 1 to the "ramp off" phase.

        Returns
        -------
        numpy.ndarray
            1D array containing the target labels for the experimental data.
        '''
        target = np.empty([self.n_subjects, self.n_subject_samples])

        for i in range(self.n_subjects):
            target[i, ::2] = 0
            target[i, 1::2] = 1

        # reshape without extra dimension for subject & subject_sample
        target = np.reshape(target, (self.n_samples))

        return target


def main(n_subjects_train=100, n_subjects_test=20, n_trials=20, plot=True):
    '''
    This main function simulates experimental dummy data, splits it into training and test sets, saves the data,
    and optionally saves a plot of a single trial.

    Parameters
    ----------
    n_subjects_train : int, optional
        The number of subjects to be used in the training set. Default is 100.
    n_subjects_test : int, optional
        The number of subjects to be used in the test set. Default is 20.
    n_trials : int, optional
        The number of trials to simulate for each subject. Default is 20.
    plot : bool, optional
        If True, a trial will be plotted. Default is True.

    Notes
    -----
    The function first simulates the data for all subjects and trials. It then splits the data into
    training and test sets, based on the number of subjects specified for each set. The data, target, 
    and groups are saved to the 'dummy' directory in the current working directory. If the 'plot' 
    parameter is set to True, the function will also plot the ECG, RSP, and SCL signals for a single trial.
    '''

    n_subjects = n_subjects_train + n_subjects_test
    experiment = SimulateExperiment(n_subjects=n_subjects, n_trials=n_trials)

    data = experiment.data[:n_subjects_train*n_trials*2]
    target = experiment.target[:n_subjects_train*n_trials*2]
    groups = experiment.groups[:n_subjects_train*n_trials*2]

    data_test = experiment.data[n_subjects_train*n_trials*2:]
    target_test = experiment.target[n_subjects_train*n_trials*2:]

    print(f"{data.shape = }\n{target.shape = }\n{groups.shape = }\n\n{data_test.shape = }\n{target_test.shape =}")
  
    # Save data
    PROJECT_DIR = Path.cwd()
    DATA_DIR = PROJECT_DIR / 'data'
    DUMMY_DIR = DATA_DIR / 'dummy'
    DUMMY_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DUMMY_DIR / "data.npy", data)
    np.save(DUMMY_DIR / "target.npy", target)
    np.save(DUMMY_DIR / "groups.npy", groups)
    np.save(DUMMY_DIR / "data_test.npy", data_test)
    np.save(DUMMY_DIR / "target_test.npy", target_test)

    # Visualise 1 trial
    if plot:
        trial = 190
        data_plot = pd.DataFrame({
            "ECG": data[trial, :, 0],
            "RSP": data[trial, :, 1],
            "SCL": data[trial, :, 2]})
        nk.signal_plot(data_plot, subplots=True)
        plt.savefig(DUMMY_DIR / "_single_dummy_trial.png")


if __name__ == '__main__':
    main()




# Code from make_dataset.py:

# import click
# import logging
# from dotenv import find_dotenv, load_dotenv
# import numpy as np
# from numpy.random import randint
# import pandas as pd
# import matplotlib.pyplot as plt
# import neurokit2 as nk
# import random

# # Include your simulation classes here

# # ...

# @click.command()
# @click.argument('output_filepath', type=click.Path())
# def main(output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')

#     # Instantiate your simulation class
#     experiment = SimulateExperiment(n_subjects=4, n_trials=3)

#     # Get the simulated data and target
#     data = experiment.data
#     target = experiment.target

#     # Save the data and target to the output file path
#     np.savez(output_filepath, data=data, target=target)

