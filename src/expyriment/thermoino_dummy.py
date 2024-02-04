import math
import logging
import time

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class ThermoinoDummy:
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
        self.connected = False

        logger.warning("+++ RUNNING IN DUMMY MODE +++")

    def connect(self):
        self.connected = True
        logger.info("Connection established.")

    def close(self):
        self.connected = False
        logger.debug("Connection closed.")

    def _send_command(self, *args):
        return "Dummy response."

    def diag(self):
        output = self._send_command('DIAG\n')
        logger.info("Basic diagnostic information: %s.", output)

    def trigger(self):
        self._send_command('START\n')
        logger.debug("Triggered.")

    def set_temp(self, temp_target):
        move_time_us = round(((temp_target - self.temp) / self.mms_rate_of_rise) * 1e6)
        duration_ms = math.ceil(abs(move_time_us / 1e3))
        self.temp = temp_target
        success = True
        output = "OK_MOVE_SLOW"
        logger.info("Change temperature to %s°C in %s s (%s).", temp_target, duration_ms / 1000, output)
        return (duration_ms, success)

    def wait(self, duration):
        time.sleep(duration)


class ThermoinoComplexTimeCoursesDummy(ThermoinoDummy):
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

    def create_ctc(self, *args, **kwargs):
        # Implement the logic for creating CTC based on temp_course and sample_rate
        # For dummy purposes, simply set self.ctc to a simulated value
        self.ctc = [40] * 100  # Example simulated CTC


    def load_ctc(self, debug=False):
        # Simulate loading CTC into the device
        logger.debug("Loading CTC into Thermoino (dummy).")
        return self._send_command('LOADCTC\n', debug)

    def query_ctc(self, queryLvl, statAbort):
        return self._send_command(f'QUERYCTC;{queryLvl};{statAbort}\n')
    
    def prep_ctc(self):
        prep_duration = 1000
        logger.debug("Prepping CTC (dummy).")
        return (self, prep_duration)
 
    def exec_ctc(self):
        exec_duration = self.bin_size_ms * len(self.ctc) / 1000
        return (self, exec_duration)

    def flush_ctc(self):
        self.ctc = None
        self._send_command('FLUSHCTC\n')

    