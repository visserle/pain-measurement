import itertools
import logging
from datetime import datetime

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class RemoteControliMotionsDummy:
    """
    A dummy version of the RemoteControliMotions class for testing and debugging.
    This class mimics the behavior of the RemoteControliMotions without requiring an actual connection to iMotions.
    """

    def __init__(self, study, participant_info: dict):
        self.study = study
        self.participant_info = participant_info
        self.connected = None
        logger.warning("+++ iMotions running in dummy mode +++")

    def _send_and_receive(self, query):
        return "OK"

    def _check_status(self):
        return 0  # Simulating iMotions is ready

    def connect(self):
        self.connected = True
        logger.info("Ready for remote control.")

    def start_study(self, mode="NormalPrompt"):
        if self.connected:
            logger.info(
                "Started recording participant %s (%s)", self.participant_info["id"], self.study
            )

    def end_study(self):
        if self.connected:
            logger.info(
                "Stopped recording participant %s (%s)", self.participant_info["id"], self.study
            )

    def abort_study(self):
        if self.connected:
            logger.info(
                "Aborted recording participant %s (%s)", self.participant_info["id"], self.study
            )

    def export_data(self, path):
        logger.info("Exported data for participant %s to %s", self.participant_info["id"], path)

    def close(self):
        self.connected = False
        logger.info("Closed remote control connection.")


class EventRecievingiMotionsDummy:
    """
    A dummy version of the EventRecievingiMotions class for testing and debugging.
    This class mimics the behavior of the EventRecievingiMotions without requiring an actual connection to iMotions.
    """

    seed_cycles = {}
    prep_cycle = itertools.cycle(
        ["M;2;;;thermode_ramp_on;start;D;\r\n", "M;2;;;thermode_ramp_off;end;D;\r\n"]
    )

    def __init__(self):
        self.connected = False
        logger.warning("+++ iMotions Event Receiver running in dummy mode +++")

    @property
    def time_stamp(self):
        return datetime.utcnow().isoformat(sep=" ", timespec="milliseconds")

    def connect(self):
        self.connected = True
        logger.info("Ready for event receiving.")

    def _send_message(self, message):
        pass

    def send_marker(self, marker_name, value):
        if self.connected:
            logger.info("Received marker %s: %s", marker_name, value)

    def send_marker_with_time_stamp(self, marker_name):
        if self.connected:
            logger.info("Received the marker %s at %s", marker_name, self.time_stamp)

    def send_stimulus_markers(self, seed):
        """
        Alternates between generating 'start of seed' and 'end of seed' stimulus markers for each unique seed value.

        This function creates and maintains a separate cycling state for each seed, ensuring that each call for a
        particular seed alternates between 'start' and 'end' markers. A new cycle is initialized for each new seed.

        """
        if seed not in self.seed_cycles:
            self.seed_cycles[seed] = itertools.cycle(
                [
                    f"M;2;;;heat_stimulus;{seed};S;\r\n",
                    f"M;2;;;heat_stimulus;{seed};E;\r\n",
                ]
            )
        string = next(self.seed_cycles[seed])
        logger.info("Recieved stimulus marker: %s", string)

    def send_prep_markers(self):
        logger.info("Received marker for thermode ramp on/off: %s", next(self.prep_cycle))

    def send_temperatures(self, temperature):
        if self.connected:
            logger.debug("Received temperature: %s", temperature)

    def send_ratings(self, rating):
        if self.connected:
            logger.debug("Received rating: %s", rating)
            
    def send_data(self, temperature, rating, debug=True):
        """
        Send temperature and rating data to iMotions in one go. 
        """
        # imotions_data = f"E;1;CustomCurves;1;;;;CustomCurves;{temperature};{rating}\r\n"
        if debug:
            logger.debug("Received temperature: %s, rating: %s.", temperature, rating)

    def send_event_x_y(self, x, y):
        if self.connected:
            logger.debug("Received X, Y event: X=%s, Y=%s", x, y)

    def close(self):
        self.connected = False
        logger.info("iMotions (dummy) connection for event receiving closed.")
