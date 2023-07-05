# work in progress

import re
import serial
import serial.tools.list_ports
import time
import numpy as np


def list_com_ports():
    """Lists all serial ports"""
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No serial ports found.")
    for port, desc, hwid in sorted(ports):
        print(f"{port}: {desc} [{hwid}]")


class Thermoino:
    """  
    The `Thermoino` class facilitates communication with the Thermoino.
    
    The class provides methods to initialize the device, set target temperature, 
    create and load complex temperature courses (ctc), and execute these courses.
    
    It is roughly based on the MATLAB script UseThermoino.m.
    
    Attributes
    ----------
    port : `str`
        The serial port that the Arduino is connected to.
    baud_rate : `int`
        The baud rate for the serial communication. Default is 115200.
    s_max_wait : `int`
        Maximum wait time in seconds. Default is 5.
    error_msg : `list` of `str`
        List of possible error messages.
    temp_baseline : `int`
        Baseline temperature in degree Celsius. Default is 32. 
        Has to be set up in MMS as well.
    temp_course_duration : `int`
        Duration of the temperature course in seconds.
    temp_course_resampled : `np.array`
        Resampled temperature course used to calculate the ctc.
    ctc : `numpy.array`
        The resampled, differentiated, binned temperature course to be loaded into the Thermoino.
    rate_of_rise : `int`
        Rate of rise of temperature in degree Celsius per second. Default is 10 for Pathways. 
        For TAS 2 it should be 13. For CHEPS something over 50 (ask Björn).
        Has to be set up in MMS as well.
    debug : `bool`
        If True, debug information will be printed. Default is True.
    ser : `serial.Serial`
        Serial object for communication with the Thermoino.
    bin_size_ms : `int`
        Bin size in milliseconds for the complex temperature course.
        
    Methods
    -------
    init():
        Initialize the Arduino.
    close():
        Close the serial connection.
    diag():
        Get basic diagnostic information from the Arduino.
    trigger():
        Trigger MMS to get ready for action.
    set_temp(temp_target):
        Set a target temperature on the Thermoino.
    init_ctc(bin_size_ms):
        Initialize a complex temperature course (ctc) on the Thermoino by sending the bin size only.
    create_ctc(temp_course, sample_rate = 100, rate_of_rise = None):
        Create a ctc based on the provided temperature course and the sample rate.
    load_ctc(debug = False):
        Load the created ctc into the Thermoino.
    exec_ctc():
        Execute the loaded ctc on the Thermoino.
        
    New stuff
    -----------
    - create_ctc, where you load your temperature course [°C/s] with sampling rate and it returns the ctc (a differentiated, binned temperature course)
    
    Examples
    --------
    ````python
    import time
    from thermoino import Thermoino, list_com_ports
    list_com_ports() # list all available serial ports
    port = "COM3"
    luigi = Thermoino(port=port)
	
    # Use luigi for complex temperature courses:
    luigi.init()
    luigi.trigger()
    luigi.init_ctc(bin_size_ms=500)
    ror, ctc = luigi.create_ctc(
        temp_course = stimuli.wave, 
        sample_rate = stimuli.sample_rate, 
        rate_of_rise = 13)
    luigi.load_ctc()
    time.sleep(1)
    luigi.exec_ctc() # TODO: find out if this works without sleep
    luigi.close()
    
    # Use luigi to set temperatures:
    luigi.init()
    luigi.trigger()
    luigi.set_temp(42)
    time.sleep(4)
    luigi.set_temp(30)
    luigi.close()
    ````

    TODO
    ----
    - Add error information to the error messages. (Note: Matlab counts from 1)
    - Add success messages (from Christian's github)
    - Add methods await_status and read_buffer and replace time.sleep
    - Add debug option to hide print statements
    - Fix query function
    - etc.
    
    """
    
    port = None
    baud_rate = 115200
    s_max_wait = 5
    error_msg = [
        'ERR_NO_PARAM', 'ERR_CMD_NOT_FOUND', 'ERR_ctc_BIN_WIDTH',
        'ERR_ctc_PULSE_WIDTH', 'ERR_ctc_NOT_INIT', 'ERR_ctc_FULL',
        'ERR_ctc_EMPTY', 'ERR_SHOCK_RANGE', 'ERR_SHOCK_ISI',
        'ERR_BUSY', 'ERR_DEBUG_RANGE']

    def __init__(self, port, temp_baseline = 32, rate_of_rise = 10, debug = True):
        """
        Constructs a Thermoino object.
        
        Parameters
        ----------
        port : `str`
            The serial port to which the Thermoino device is connected.
        temp_baseline : `int`, optional
            Baseline temperature in °C. Default is 32. Important: Has to be the same as in MMS.
        rate_of_rise : `int`
            Rate of rise of temperature in degree Celsius per second. Default is 10 for Pathways. 
            For TAS 2 it should be 13. For CHEPS something over 50 (ask Björn).
            May get changed later with create_ctc(), if not specified here.
        debug : `bool`, optional
            If True, debug information will be printed. Default is True.
        """
        Thermoino.port = port
        self.temp_baseline = temp_baseline
        self.temp = temp_baseline # temperature we start with -> will be updated
        self.rate_of_rise = rate_of_rise
        self.debug = debug

    def init(self):
        """
        Initialize the Thermoino. 
        This method establishes a serial connection to the device and waits for it to boot up.
        
        Returns
        -------
        `serial.Serial`
            The serial object for communication with the device.
        """
        self.ser = serial.Serial(self.port, self.baud_rate)   
        time.sleep(1)
        return self.ser

    def close(self):
        """
        Close the serial connection.
        This method should be called manually to close the connection when it's no longer needed.
        As this class is not a context manager, the connection is not closed automatically when an error occurs.
        """
        self.ser.close()

    def diag(self):
        """
        Send a 'DIAG' command to the Arduino to get basic diagnostic information.
        Or even more? (ask Björn)
        """
        self.ser.write(b'DIAG\n')
        print("Sent 'DIAG' (.diag) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'DIAG': {output}")
        
    def trigger(self):
        """
        Trigger MMS to get ready for action.
        """
        self.ser.write(b'START\n')
        print("Sent 'START' (.trigger) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'START': {output}")

    def set_temp(self, temp_target):
        """
        Set a target temperature on the Thermoino device. 
        Most common function in standard use.
        It is based on the 'MOVE' command, which is the most basic Thermoino command.
        
        Parameters
        ----------
        temp_target : `int`
            The target temperature in degree Celsius.
        """
        move_time_us = round(((temp_target - self.temp) / self.rate_of_rise) * 1e6)
        self.ser.write(f'MOVE;{move_time_us}\n'.encode())
        print("Sent 'MOVE' (.set_temp) to Arduino")
        print(f"t = {move_time_us / 1e6}")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'MOVE': {output}")
        self.temp = temp_target

    def init_ctc(self, bin_size_ms):
        """
        Initialize a complex temperature course (ctc) on the Thermoino device
        by defining the bin size only. This has to be done before loading the ctc 
        into the Arduino (load_ctc).
        
        Parameters
        ----------
        bin_size_ms : `int`
            The bin size in milliseconds for the ctc.
        
        Returns
        -------
        `int`
            The bin size in milliseconds for the ctc.
        """
        self.ser.write(f'INITCTC;{bin_size_ms}\n'.encode())
        print("Sent 'INITCTC' (.init_ctc) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'INITCTC': {output}")
        self.bin_size_ms = bin_size_ms

        return self.bin_size_ms
    
    def create_ctc(self, temp_course, sample_rate, rate_of_rise): 
        """
        Create a complex temperature course (ctc) based on the provided temperature course, the sample rate and the rate of rise.
        A ctc is a differentiated, binned temperature course. The rate of rise either has to be provided or will be determined from the temperature course.
        In the latter case, an "optimal" rate of rise will be returned, as the lower the rate of rise is, the more precise the temperature control via the thermode.

        On the x-axis, the time course is defined in bin_size_ms. 
        On the y-axis, the amount of time for opening the thermode in a bin is defined in ms.
        
        Parameters
        ----------
        temp_course : `numpy.ndarray`
            The temperature course in degree Celsius per second.
        sample_rate : `int`, optional
            Sample rate in Hz. Default is 100.
        rate_of_rise : `int`, optional
            Rate of rise of temperature in degree Celsius per second. 
            Most common value is the same as in the Thermoino object as it has to be set up in MMS as well.
            If None provided, it will be determined from the temperature course.
        
        Returns
        -------
        `int`
            The rate of rise of temperature in degree Celsius per second.
        `numpy.ndarray`
            The created ctc. 
            Note: The length of the ctc is one less than the length of the temperature course because of np.diff. This is accounted for in the Arduino code.
        """

        self.temp_course_duration = temp_course.shape[0] / sample_rate
        # Resample the temperature course according to the bin size:
        # i.e. for a 100 s stimuli with a bin size of 500 ms we'd need 200 bins á 500 ms
        temp_course_resampled = temp_course[::int(sample_rate / (1000 / self.bin_size_ms))]
        self.temp_course_resampled = temp_course_resampled
        temp_course_resampled_diff = np.diff(temp_course_resampled)

        if not rate_of_rise:
            # determine optimal rate of rise (has to be updated in MMS accordingly)
            rate_of_rise = max(temp_course_resampled_diff * (1000 / self.bin_size_ms))
            rate_of_rise = np.ceil(rate_of_rise * 10) / 10  # round up to .1°C precision
            rate_of_rise_ms = rate_of_rise / 1e3
        else:
            rate_of_rise_ms = self.rate_of_rise / 1e3
        
        # Update the rate of rise
        self.rate_of_rise = rate_of_rise

        # scale to rate_of_rise (in milliseconds)
        temp_course_diff_binned = temp_course_resampled_diff / rate_of_rise_ms
        # Thermoino only accepts integers
        temp_course_diff_binned = np.ceil(temp_course_diff_binned).astype(int)
        
        self.ctc = temp_course_diff_binned
        return self.rate_of_rise, self.ctc

    def load_ctc(self, debug = False):      
        """
        Load the created ctc into the Thermoino device by sending single bins in a for loop to the Thermoino.
        Note: The maximum length of the array is 2500. If you want longer stimuli, you have to use a larger bin size.
        (Also keep in mind the 10 min limit of MMS.)
        TODO: add await_status
        
        Parameters
        ----------
        debug : `bool`, optional
            If True, debug information will be printed. Default is False.
        """

        for i in range(len(self.ctc)):
            self.ser.write(f'LOADCTC;{self.ctc[i]}\n'.encode())
            # workaround: time.sleep after every iteration,
            # not using await_status at the moment
            time.sleep(0.05)
            if debug:
                output = self.ser.readline().decode('ascii')
                print(f"Received output from 'LOADCTC': {output}")
        print("Sent 'LOADCTC' (.load_ctc) to Arduino")        

    def query_ctc(self, queryLvl, statAbort):
        """
        Query information about the complex temperature course (ctc) on the Thermoino device.

        This method sends a 'QUERYCTC' command to the device. Depending on the query level (`queryLvl`), 
        different types of information are returned, e.g. ctcStatus, ctcBinSize, 
        ctc length, the ctc itself.

        Parameters
        ----------
        queryLvl : `int`
            The query level. 
            Level 1 returns only the ctc status and verbose status description. 
            Level 2 returns additional information, including ctc bin size, ctc length, ctc execution flag, 
            and the full ctc (which can take some time to transfer).
        statAbort : `bool`
            If True and the ctc status is 0, an error is raised and execution is stopped.

        Returns
        -------
        `str`
            The output from the Thermoino device.
        """
        self.ser.write(f'QUERYCTC;{queryLvl};{statAbort}\n'.encode())
        print("Sent 'QUERYCTC' (.query_ctc) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'QUERYCTC': {output}")
        return output

    def exec_ctc(self):
        """
        Execute the loaded ctc on the Thermoino device. This results in sending thermode pulses.
        
        TODO: 
        
        - add msOffs.msOffs requires sum(diff(tC)) to estimate temp at ctc offset;
        st is optional and includes sum(diff(tC)) and new temp
        """
        # Always start fom the baseline that is also specified in MMS
        if not self.temp == self.temp_baseline:
            self.set_temp(self.temp_baseline)
            time.sleep(0.3)
        self.ser.write(f'EXECCTC\n'.encode())
        print("Sent 'EXECCTC' (.exec_ctc) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'EXECCTC': {output}")
        self.temp = round(self.temp_course_resampled[-1],2)
        print(f"Set temperature to {self.temp}°C after the ctc was executed.")

    def flush_ctc(self):
        """
        Reset or delete all complex temperature course (ctc) information on the Thermoino device.

        This method sends a 'FLUSHCTC' command to the device. It can be called individually, but it is 
        also automatically called by the `init_ctc` method.

        Returns
        -------
        `str`
            The output from the Thermoino device after sending the 'FLUSHCTC' command.
        """
        self.ser.write(f'FLUSHCTC\n'.encode())
        print("Sent 'FLUSHCTC' (.flush_ctc) to Arduino")
        output = self.ser.readline().decode('ascii')
        print(f"Received output from 'FLUSHCTC': {output}")
		
		
if __name__ == "__main__":
    pass