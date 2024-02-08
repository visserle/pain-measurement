"""Module for communication with the Thermoino (a composite of Thermode & Arduino)."""
# work in progress

# TODO
# - Fix query function

import logging
import math
import time
import warnings
from enum import Enum

import numpy as np
import serial
import serial.tools.list_ports

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def list_com_ports():
    """List all serial ports"""
    ports = serial.tools.list_ports.comports()
    output = []
    if len(ports) == 0:
        output.append("No serial ports found.")
    for port, desc, hwid in sorted(ports):
        output.append(f"{port}: {desc} [{hwid}]")
    return "\n".join(output)


class ErrorCodes(Enum):
    """Error codes as defined in the Thermoino code (Thermode_PWM.ino)"""

    ERR_NULL = 0
    ERR_NO_PARAM = -1
    ERR_CMD_NOT_FOUND = -2
    ERR_CTC_BIN_WIDTH = -3
    ERR_CTC_PULSE_WIDTH = -4
    ERR_CTC_NOT_INIT = -5
    ERR_CTC_FULL = -6
    ERR_CTC_EMPTY = -7
    ERR_SHOCK_RANGE = -8
    ERR_SHOCK_ISI = -9
    ERR_BUSY = -10
    ERR_DEBUG_RANGE = -11


class OkCodes(Enum):
    """OK codes as defined in the Thermoino code (Thermode_PWM.ino)"""

    OK_NULL = 0
    OK = 1
    OK_READY = 2
    OK_MOVE_SLOW = 3
    OK_MOVE_PREC = 4


class Thermoino:
    """
    The `Thermoino` class facilitates communication with the Thermoino (a composite of
    Thermode and Arduino).

    The class provides methods to initialize the device and set target temperatures.

    For the most part, it is based on the MATLAB script UseThermoino.m.

    Attributes
    ----------
    PORT : `str`
        The serial port that the Thermoino is connected to.
    BAUD_RATE : `int`
        The baud rate for the serial communication. Default is 115200.
    ser : `serial.Serial`
        Serial object for communication with the Thermoino.
    temp : `int`
        Current (calculated) temperature [°C]. Starts at the baseline temperature.
    mms_baseline : `int`
        Baseline temperature [°C].
        It has to be the same as in the MMS program.
    mms_rate_of_rise : `int`
        Rate of rise of temperature [°C/s]. It has to be the same as in the MMS program.
        For Pathways 10 is standard. For TAS 2 it is 13. For CHEPS something over 50 (ask Björn).
        For normal temperature plateaus a higher rate of rise is recommended (faster);
        for complex temperature courses a lower rate of rise is recommended (more precise).

    Methods
    -------
    connect():
        Connect to the Thermoino via serial connection.
    close():
        Close the serial connection.
    diag():
        Get basic diagnostic information from the Thermoino.
    trigger():
        Trigger MMS to get ready for action.
    set_temp(temp_target):
        Set a target temperature on the Thermoino.
    sleep(duration):
        sleep for a given duration in seconds.

    New stuff
    -----------
    - renamed init() to connect to be consistent with python naming conventions

    Examples
    --------
    ```python
    from thermoino import Thermoino, list_com_ports

    # List all available serial ports
    from src.expyriment.thermoino import Thermoino, list_com_ports

    # List all available serial ports
    list_com_ports()

    port = "COM7"
    thermoino = Thermoino(
        port=port,
        mms_baseline=28,  # has to be the same as in MMS
        mms_rate_of_rise=10,  # has to be the same as in MMS
    )

    # Use thermoino to set temperatures:
    thermoino.connect()
    thermoino.trigger()
    time_to_ramp_up, _ = thermoino.set_temp(42)
    thermoino.sleep(duration_ms=time_to_ramp_up)
    thermoino.sleep(8000)  # 8 s plateau
    time_to_ramp_down, _ = thermoino.set_temp(28)
    thermoino.sleep(time_to_ramp_down)
    thermoino.close()
    ```
    """

    BAUD_RATE = 115200

    def __init__(self, port, mms_baseline, mms_rate_of_rise, dummy=False):
        """
        Constructs a Thermoino object.

        Parameters
        ----------
        port : `str`
            The serial port to which the Thermoino device is connected.
        mms_baseline : `int`, optional
            Baseline temperature in °C. It has to be the same as in the MMS program.
        mms_rate_of_rise : `int`
            Rate of rise of temperature in °C/s. It has to be the same as in the MMS program.
            For Pathways 10 is standard. For TAS 2 it is 13. For CHEPS something over 50 (ask Björn).
            It can be changed class-wide by calucalting an adjusted rate of rise in the create_ctc method.
        """
        self.PORT = port
        self.ser = None  # will be set to the serial object in connect()
        self.mms_baseline = mms_baseline
        self.temp = mms_baseline  # will get continuous class-internal updates to match the temperature of the Thermode
        self.mms_rate_of_rise = mms_rate_of_rise
        self.dummy = dummy
        if self.dummy:
            logger.critical("+++ RUNNING IN DUMMY MODE +++")

    def connect(self):
        """
        Connect to the Thermoino device.
        This method establishes a serial connection to the device and waits for it to boot up.

        Returns
        -------
        `serial.Serial`
            The serial object for communication with the device.
        """
        try:
            self.ser = serial.Serial(self.PORT, self.BAUD_RATE) if not self.dummy else DummySerial()
            logger.info("Connection established.")
            time.sleep(1) if not self.dummy else None
        except serial.SerialException:
            logger.error("Connection failed @ %s.", self.PORT)
            logger.info(f"Available serial ports are:\n{list_com_ports()}")
            raise serial.SerialException(f"Thermoino connection failed @ {self.PORT}.")

    def close(self):
        """
        Close the serial connection.

        This method should be called manually to close the connection when it's no longer needed.
        As the `Thermoino` class is not a context manager, the connection is not closed automatically.
        """
        self.ser.close()
        logger.info("Connection closed.")
        time.sleep(1)

    def _send_command(self, command, get_response=True):
        """
        Send a command to the Thermoino, read the response, decode it,
        and then pass the numeric response code to the _handle_response function.
        """
        self.ser.write(command.encode())  # encode to bytes
        if get_response:
            response = self.ser.readline()
            try:
                decoded_response = response.decode("ascii").strip()
            except UnicodeDecodeError:
                logger.error("Thermoino response could not be decoded: %s", response)
                decoded_response = None
            return self._handle_response(decoded_response)
        return None

    def _handle_response(self, decoded_response):
        """
        Take the decoded response from _send_command, determine if it's an error or success
        code based on whether it's less than 0, and convert it to the corresponding enum value.
        """

        def _is_integer(s):
            """Check if a string is an integer, .isdigit() does not work for negative numbers."""
            try:
                int(s)
                return True
            except ValueError:
                return False

        if not _is_integer(decoded_response):
            return decoded_response  # e.g. when using diag

        decoded_response = int(decoded_response)
        if decoded_response < 0:
            response = ErrorCodes(decoded_response).name
        else:
            response = OkCodes(decoded_response).name
        return response

    def diag(self):
        """
        Send a 'DIAG' command to the Thermoino to get basic diagnostic information.
        """
        output = self._send_command("DIAG\n")
        logger.info("Diagnostic information: %s.", output)

    def trigger(self):
        """
        Trigger MMS to get ready for action.
        """
        output = self._send_command("START\n")
        if output in OkCodes.__members__:
            logger.debug("Triggered.")
        elif output in ErrorCodes.__members__:
            logger.critical("Triggering failed: %s.", output)

    def set_temp(self, temp_target):
        """
        Set a target temperature on the Thermoino device.
        Most common function in standard use.
        It is based on the 'MOVE' command, which is the most basic Thermoino command.

        Parameters
        ----------
        temp_target : `int`
            The target temperature in degree Celsius.

        Notes
        -----
        The command MOVE does ramp up (positive numbers) or down (negative numbers) for x microseconds (move_time_us).

        Returns
        -------
        tuple
            (float, bool) - float for the duration in seconds for the temperature change, bool for success
        """

        move_time_us = round(((temp_target - self.temp) / self.mms_rate_of_rise) * 1e6)
        output = self._send_command(f"MOVE;{move_time_us}\n")
        duration = math.ceil(abs(move_time_us / 1e6))
        if output in OkCodes.__members__:
            # Update the current temperature
            self.temp = temp_target
            logger.info(
                "Change temperature to %s°C in %s s: %s.", temp_target, round(duration, 2), output
            )
            success = True
        elif output in ErrorCodes.__members__:
            logger.error("Setting temperature to %s°C failed: %s.", temp_target, output)
            success = False
        return (duration, success)

    def sleep(self, duration):
        """
        - NOT RECOMMENDED -

        FIXME: Sleep for a given duration in seconds.
        This function delays the execution in Python for a given number of seconds.

        It should not be used in a experiment (e.g. for continuous ratings) as this function blocks the execution
        of anything else.

        Parameters
        ----------
        duration : float
            The duration to sleep in seconds.
        """
        logger.warning("Sleeping for %s s using time.sleep.", duration)
        time.sleep(duration)


class ThermoinoComplexTimeCourses(Thermoino):
    """
    The `ThermoinoComplexTimeCourses` class facilitates communication with the Thermoino for complex temperature courses (CTC).

    The class inherits from `Thermoino` and provides methods to initialize the device, set target temperatures,
    create and load complex temperature courses (CTC) on the Thermoino, and execute these courses.

    For the most part, it is based on the MATLAB script UseThermoino.m.


    Attributes
    ----------
    PORT : `str`
        The serial port that the Thermoino is connected to.
    BAUD_RATE : `int`
        The baud rate for the serial communication. Default is 115200.
    ser : `serial.Serial`
        Serial object for communication with the Thermoino.
    temp : `int`
        Current (calculated) temperature [°C]. Starts at the MMS baseline temperature.
    mms_baseline : `int`
        Baseline temperature [°C].
        It has to be the same as in the MMS program.
    temp_course_duration : `int`
        Duration of the temperature course [s].
    temp_course_resampled : `np.array`
        Resampled temperature course (based on bin_size_ms) used to calculate the CTC.
    ctc : `numpy.array`
        The resampled, differentiated, binned temperature course to be loaded into the Thermoino.
    mms_rate_of_rise : `int`
        Rate of rise of temperature [°C/s]. It has to be the same as in the MMS program.
        For Pathways 10 is standard. For TAS 2 it is 13. For CHEPS something over 50 (ask Björn).
        For normal temperature plateaus a higher rate of rise is recommended (faster);
        for complex temperature courses a lower rate of rise is recommended (more precise).
    bin_size_ms : `int`
        Bin size in milliseconds for the complex temperature course.

    Methods
    -------
    connect():
        Connect to the Thermoino via serial connection.
    close():
        Close the serial connection.
    diag():
        Get basic diagnostic information from the Thermoino.
    trigger():
        Trigger MMS to get ready for action.
    set_temp(temp_target):
        Set a target temperature on the Thermoino.
    sleep(duration):
        sleep for a given duration in seconds (using time.sleep).
    init_ctc(bin_size_ms):
        Initialize a complex temperature course (CTC) on the Thermoino by sending the bin size only.
    create_ctc(temp_course, sample_rate, rate_of_rise_option = "mms_program"):
        Create a CTC based on the provided temperature course and the sample rate.
    load_ctc(debug = False):
        Load the created CTC into the Thermoino.
    query_ctc(queryLvl, statAbort):
        Query the Thermoino for information about the CTC.
    prep_ctc():
        Prepare the starting temperature of the CTC.
    exec_ctc():
        Execute the loaded CTC on the Thermoino.
    flush_ctc()
        Reset CTC information on the Thermoino. This has to be done before loading a new CTC.

    New stuff
    -----------
    - renamed init() to connect() to be consistent with python naming conventions
    - create_ctc(), where you load your temperature course with sampling rate and it returns the CTC (a resampled, differentiated, binned temperature course)
    - prep_ctc() which prepares the starting temperature for execution of the CTC

    Examples
    --------
    ````python
    from thermoino import ThermoinoComplexTimeCourses, list_com_ports

    # List all available serial ports
    list_com_ports()

    port = "COM7"
    thermoino = ThermoinoComplexTimeCourses(
        port=port,
        mms_baseline=28,  # has to be the same as in MMS
        mms_rate_of_rise=10,  # has to be the same as in MMS
    )

    # Use thermoino for complex temperature courses:
    thermoino.connect()
    thermoino.init_ctc(bin_size_ms=500)
    thermoino.create_ctc(temp_course=stimulus.wave, sample_rate=stimulus.sample_rate)
    thermoino.load_ctc()
    thermoino.trigger()
    time_to_ramp_up = thermoino.prep_ctc()
    thermoino.sleep(duration_ms=time_to_ramp_up)
    time_to_exec_ctc = thermoino.exec_ctc()
    thermoino.sleep(time_to_exec_ctc)
    time_to_ramp_down, _ = thermoino.set_temp(32)
    thermoino.flush_ctc()
    thermoino.close()

    # Use thermoino to set temperatures:
    thermoino.connect()
    thermoino.trigger()
    time_to_ramp_up, _ = thermoino.set_temp(42)
    thermoino.sleep(duration_ms=time_to_ramp_up)
    thermoino.sleep(8000)  # 8 s plateau
    time_to_ramp_down, _ = thermoino.set_temp(28)
    thermoino.sleep(time_to_ramp_down)
    thermoino.close()
    ````
    """

    def __init__(self, port, mms_baseline, mms_rate_of_rise):
        super().__init__(port, mms_baseline, mms_rate_of_rise)
        logger.info("Thermoino for complex time courses initialized.")
        self.bin_size_ms = None
        self.temp_course = None
        self.temp_course_duration = None
        self.temp_course_start = None
        self.temp_course_end = None
        self.ctc = None

    def init_ctc(self, bin_size_ms):
        """
        Initialize a complex temperature course (CTC) on the Thermoino device
        by firstly defining the bin size in milliseconds. This has to be done before loading the CTC
        into the Thermoino (load_ctc).

        This function also reset all ctc information stored on the Thermoino device.
        """
        output = self._send_command(f"INITCTC;{bin_size_ms}\n")
        if output in OkCodes.__members__:
            logger.debug("Complex temperature course (CTC)) initialized.")
            self.bin_size_ms = bin_size_ms
        elif output in ErrorCodes.__members__:
            logger.error("Initializing complex temperature course (CTC) failed: %s.", output)

    def create_ctc(self, temp_course, sample_rate, rate_of_rise_option="mms_program"):
        """
        Create a complex temperature course (CTC) based on the provided temperature course, the sample rate.
        A CTC is a differentiated, binned temperature course. The rate of rise either is either the
        same as in the `Thermoino` instance or will be determined from the temperature course.
        In the latter case, an "optimal" rate of rise will be returned, as the lower the rate of rise is,
        the more precise the temperature control via the thermode.

        Either way, the rate of rise must be the same as specified in the MMS program.

        On the x-axis, the time course is defined in bin_size_ms.
        On the y-axis, the amount of time for opening the thermode in a bin is defined in ms.

        Parameters
        ----------
        temp_course : `numpy.ndarray`
            The temperature course [°C] in s.
        sample_rate : `int`
            Sample rate in Hz.
        rate_of_rise_option : `str`, optional
            Rate of rise of temperature in degree Celsius per second.
            Default is "mms_program", which uses the same rate of rise as specified in the Thermoino object.
            If "adjusted" provided, an "optimal" rate of rise will be determined from the temperature course.
            (The lower the rate of rise is, the more precise the temperature control via the thermode.)

        Side effects
        ------------
        Creates / modifies the following attributes (self.):\n
        `temp_course_duration` : `int`
            Duration of the temperature course [s].
        `temp_course_resampled` : `np.array` # TODO update, we only need start and end FIXME all doc strings
            Resampled temperature course (based on bin_size_ms) used to calculate the CTC.
        `rate_of_rise` : `int`
            The rate of rise of temperature [°C/s].
            Mofidied if rate of rise option is "adjusted".
        `ctc` : `numpy.array`
            The created CTC.
        """
        self.temp_course_duration = temp_course.shape[0] / sample_rate
        temp_course_resampled = (
            pd.DataFrame(
                {"temp": temp_course},
                index=pd.to_timedelta(np.arange(len(temp_course)) / sample_rate, unit="s"),
            )
            .resample(f"{self.bin_size_ms}ms")
            .ffill()
            .to_numpy()
            .flatten()
        )
        self.temp_course_start = temp_course_resampled[0]
        self.temp_course_end = temp_course_resampled[-1]
        # TODO: diff or gradient -> Christian
        temp_course_resampled_diff = np.diff(temp_course_resampled)
        mms_rate_of_rise_ms = self.mms_rate_of_rise / 1e3
        # scale to mms_rate_of_rise (in milliseconds)
        temp_course_resampled_diff_binned = temp_course_resampled_diff / mms_rate_of_rise_ms
        # Thermoino only accepts integers
        temp_course_resampled_diff_binned = np.round(temp_course_resampled_diff_binned).astype(int)
        self.ctc = temp_course_resampled_diff_binned
        logger.debug(
            "Complex temperature course (CTC) created with %s bins of %s ms.",
            len(self.ctc),
            self.bin_size_ms,
        )

    def load_ctc(self, debug=False):
        """
        Load the created CTC into the Thermoino device by sending single bins in a for loop to the Thermoino.
        The maximum length to store on the Thermoino is 2500. If you want longer stimuli, you could use a larger bin size.
        (The max bin size is 500 ms, also keep in mind the 10 min limit of MMS.)

        Parameters
        ----------
        debug : `bool`, optional
            If True, debug information for every bin. Default is False for performance.
        """
        logger.debug("Complex temperature course (CTC) loading started ...")
        for idx, i in enumerate(self.ctc):
            output = self._send_command(f"LOADCTC;{i}\n")

            if output in ErrorCodes.__members__:
                logger.error(
                    f"Error while loading bin {idx + 1} of {len(self.ctc)}. Error code: {output}"
                )
                raise ValueError(
                    f"Error while loading bin {idx + 1} of {len(self.ctc)}. Error code: {output}"
                )
            elif debug:
                logger.debug("Bin %s of %s loaded. Response: %s.", idx + 1, len(self.ctc), output)

        logger.debug("Complex temperature course (CTC) loaded.")

    def query_ctc(self, queryLvl, statAbort):
        """
        Query information about the complex temperature course (CTC) on the Thermoino device.

        This method sends a 'QUERYCTC' command to the device. Depending on the query level (`queryLvl`),
        different types of information are returned, e.g. ctcStatus, ctcBinSize,
        CTC length, the CTC itself.

        Parameters
        ----------
        queryLvl : `int`
            The query level.
            Level 1 returns only the CTC status and verbose status description.
            Level 2 returns additional information, including CTC bin size, CTC length, CTC execution flag,
            and the full CTC (which can take some time to transfer).
        statAbort : `bool`
            If True and the CTC status is 0, an error is raised and execution is stopped.

        Returns
        -------
        `str`
            The output from the Thermoino device.
        """
        output = self._send_command(f"QUERYCTC;{queryLvl};{statAbort}\n")
        logger.info("Querying complex temperature course (CTC) information: %s.", output)

    def prep_ctc(self) -> float:
        """
        Prepare the CTC for the execution by setting the starting temperature. It returns the duration for the temperature to be reached but does not wait.

        Returns the duration in s from the set_temp function.
        """
        logger.info("Prepare the starting temperature of the complex temperature course (CTC).")
        prep_duration, success = self.set_temp(self.temp_course_start)
        if not success:
            logger.error("Preparing complex temperature course (CTC) failed.")
        return prep_duration

    def exec_ctc(self):
        """
        Execute the CTC on the Thermoino device.
        """
        if self.temp != self.temp_course_start:
            logger.error(
                "Temperature is not set at the starting temperature of the temperature course. Please run prep_ctc first."
            )
            raise ValueError(
                "Temperature is not set at the starting temperature of the temperature course. Please run prep_ctc first."
            )

        exec_duration_s = self.temp_course_duration
        output = self._send_command("EXECCTC\n")
        if output in OkCodes.__members__:
            # Update the temperature to the last temperature of the CTC
            self.temp = self.temp_course_end
            logger.info("Complex temperature course (CTC) started.")
            logger.debug("This will take %s s to finish.", round(exec_duration_s, 2))
            logger.debug("Temperature after execution: %s°C.", self.temp)
        elif output in ErrorCodes.__members__:
            logger.error("Executing complex temperature course (CTC) failed: %s.", output)
        return exec_duration_s

    def flush_ctc(self):
        """
        Reset or delete all complex temperature course (CTC) information on the Thermoino device.

        This method sends a 'FLUSHCTC' command to the device. Before loading a new CTC, the old one has to be flushed or else it will be appended.
        """
        output = self._send_command("FLUSHCTC\n")
        if output in OkCodes.__members__:
            logger.debug("Flushed complex temperature course (CTC) from memory.")
        elif output in ErrorCodes.__members__:
            logger.error("Flushing complex temperature course (CTC) failed: %s.", output)


class DummySerial:    
    def __init__(self, *args, **kwargs):
        self.response = None

    def write(self, arg):
        if "START" in arg.decode():
            self.response = "2"
        elif "MOVE" in arg.decode():
            self.response = "3"
        elif "INITCTC" in arg.decode():
            self.response = "1"
        elif "LOADCTC" in arg.decode():
            self.response = "1"
        elif "QUERYCTC" in arg.decode():
            self.response = "1"
        elif "EXECCTC" in arg.decode():
            self.response = "1"
        elif "FLUSHCTC" in arg.decode():
            self.response = "1"
        else:
            self.response = "0"

    def readline(self, *args, **kwargs):
        return self.response.encode()

    def close(self, *args, **kwargs):
        pass


def main():
    from src.log_config import configure_logging
    configure_logging(stream_level=logging.DEBUG)
    # List all available serial ports
    print(list_com_ports())

    port = "COM7"
    thermoino = Thermoino(
        port=port,
        mms_baseline=28,  # has to be the same as in MMS
        mms_rate_of_rise=10,  # has to be the same as in MMS
        dummy=True
    )

    # Use thermoino to set temperatures:
    thermoino.connect()
    thermoino.trigger()
    time_to_ramp_up, _ = thermoino.set_temp(42)
    thermoino.sleep(duration=time_to_ramp_up)
    thermoino.sleep(8)  # 8 s plateau
    time_to_ramp_down, _ = thermoino.set_temp(28)
    thermoino.sleep(time_to_ramp_down)
    thermoino.close()
    
    
if __name__ == "__main__":
    main()
