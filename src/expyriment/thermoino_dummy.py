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

        logger.warning("+++ Thermoino running in dummy mode +++")

    def connect(self):
        self.connected = True
        logger.info("Thermoino (dummy) connected")

    def close(self):
        self.connected = False
        logger.debug("Thermoino (dummy) closed")

    def _send_command(self, *args):
        return None

    def diag(self):
        return self._send_command('DIAG\n')

    def trigger(self):
        return self._send_command('START\n')

    def set_temp(self, temp_target):
        move_time_s = abs(temp_target - self.temp) / self.mms_rate_of_rise
        self.temp = temp_target
        success = True
        logging.debug("Thermoino (dummy) sets temperature to %s°C in %s seconds", temp_target, move_time_s)
        return (move_time_s, success)

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
        logger.debug("Thermoino (dummy) received the whole CTC.")
        return self._send_command('LOADCTC\n', debug)

    def query_ctc(self, queryLvl, statAbort):
        return self._send_command(f'QUERYCTC;{queryLvl};{statAbort}\n')
    
    def prep_ctc(self):
        prep_duration = 1
        logger.debug("Thermoino (dummy) is ready to execute the CTC.")
        return (self, prep_duration)
 
    def exec_ctc(self):
        exec_duration = self.bin_size_ms * len(self.ctc) / 1000
        return (self, exec_duration)

    def flush_ctc(self):
        self.ctc = None
        self._send_command('FLUSHCTC\n')

    