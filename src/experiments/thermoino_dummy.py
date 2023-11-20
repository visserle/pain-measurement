import logging
import time

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DummySerial:
    """
    A dummy serial class to simulate communication with the Thermoino device
    """
    def write(self, command):
        return

    def readline(self):
        return b'OK'

    def close(self):
        return


class Thermoino:
    """
    A dummy version of the Thermoino class for testing and debugging.
    This class rudimentarily mimics the behavior of the Thermoino without requiring an actual device.
    """

    BAUD_RATE = 115200

    def __init__(self, port, mms_baseline, mms_rate_of_rise):
        self.PORT = port
        self.mms_baseline = mms_baseline
        self.temp = mms_baseline
        self.mms_rate_of_rise = mms_rate_of_rise
        self.ser = DummySerial()
        self.connected = False

        logger.warning("+++ Thermoino running in dummy mode +++")

    def connect(self):
        self.connected = True
        logger.info("Thermoino (dummy) connected")
        return self

    def close(self):
        self.connected = False
        self.ser.close()
        logger.info("Thermoino (dummy) closed")

    def _send_command(self, command, get_response=True):
        self.ser.write(command)
        if get_response:
            response = self.ser.readline()
            return response.decode('ascii').strip()
        return None

    def diag(self):
        return self._send_command('DIAG\n')

    def trigger(self):
        return self._send_command('START\n')

    def set_temp(self, temp_target):
        move_time_s = abs(temp_target - self.temp) / self.mms_rate_of_rise
        self.temp = temp_target
        success = True
        return (self, move_time_s, success)

    def wait(self, duration):
        time.sleep(duration)
        return self


class ThermoinoComplexTimeCourses(Thermoino):
    """
    A dummy version of the ThermoinoComplexTimeCourses class for testing and debugging.
    This class rudimentarily mimics the behavior of the Thermoino without requiring an actual device.
    """

    def __init__(self, port, mms_baseline, mms_rate_of_rise):
        super().__init__(port, mms_baseline, mms_rate_of_rise)
        self.bin_size_ms = None
        self.temp_course = None
        self.temp_course_duration = None
        self.temp_course_resampled = None
        self.ctc = None

    def init_ctc(self, bin_size_ms):
        self.bin_size_ms = bin_size_ms
        return self._send_command(f'INITCTC;{bin_size_ms}\n')

    def create_ctc(self, temp_course, sample_rate, rate_of_rise_option="mms_program"):
        # Implement the logic for creating CTC based on temp_course and sample_rate
        # For dummy purposes, simply set self.ctc to a simulated value
        self.ctc = [40] * 100  # Example simulated CTC
        return self

    def load_ctc(self, debug=False):
        # Simulate loading CTC into the device
        logger.info("Thermoino (dummy) received the whole CTC.")
        return self._send_command('LOADCTC\n', debug)

    def query_ctc(self, queryLvl, statAbort):
        return self._send_command(f'QUERYCTC;{queryLvl};{statAbort}\n')
    
    def prep_ctc(self):
        prep_duration = 1
        logger.info("Thermoino (dummy) is ready to execute the CTC.")
        return (self, prep_duration)
 
    def exec_ctc(self):
        exec_duration = self.bin_size_ms * len(self.ctc) / 1000
        return (self, exec_duration)

    def flush_ctc(self):
        self.ctc = None
        self._send_command('FLUSHCTC\n')
        return self
    