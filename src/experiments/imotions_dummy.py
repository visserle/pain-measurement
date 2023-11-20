import logging
from datetime import datetime

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class RemoteControliMotions:
    """
    A dummy version of the RemoteControliMotions class for testing and debugging.
    This class mimics the behavior of the RemoteControliMotions without requiring an actual connection to iMotions.
    """
    def __init__(self,
                 study="test",
                 participant = "dummy",
                 age = 20,
                 gender = "Female"):
        self.study = study
        self.participant = participant
        self.age = age
        self.gender = gender
        self.connected = None
        logger.warning("+++ iMotions running in dummy mode +++")

    def _send_and_receive(self, query):
        return "OK"

    def _check_status(self):
        return 0  # Simulating iMotions is ready

    def connect(self):
        self.connected = True
        logger.info("iMotions (dummy) is ready for remote control.")

    def start_study(self, mode="NormalPrompt"):
        if self.connected:
            logger.info("iMotions (dummy) started study %s for participant %s", self.study, self.participant)

    def end_study(self):
        if self.connected:
            logger.info("iMotions (dummy) ended study %s for participant %s", self.study, self.participant)

    def abort_study(self):
        if self.connected:
            logger.info("iMotions (dummy) aborted study %s for participant %s", self.study, self.participant)

    def export_data(self, path):
        logger.info("iMotions (dummy) exported data for study %s for participant %s to %s", self.study, self.participant, path)

    def close(self):
        self.connected = False
        logger.info("iMotions (dummy) connection for remote control closed.")


class EventRecievingiMotions:
    """
    A dummy version of the EventRecievingiMotions class for testing and debugging.
    This class mimics the behavior of the EventRecievingiMotions without requiring an actual connection to iMotions.
    """

    def __init__(self):
        self.connected = False
        logger.warning("+++ iMotions Event Receiver running in dummy mode +++")

    @property
    def time_stamp(self):
        return datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')

    def connect(self):
        self.connected = True
        logger.info("iMotions (dummy) is ready for event receiving.")

    def _send_message(self, message):
        pass

    def start_study(self):
        if self.connected:
            logger.info("iMotions (dummy) received the marker for study start @ %s", self.time_stamp)

    def end_study(self):
        if self.connected:
            logger.info("iMotions (dummy) received the marker for study end @ %s", self.time_stamp)

    def send_marker(self, marker_name, value):
        if self.connected:
            logger.info("iMotions (dummy) received the marker %s: %s", marker_name, value)

    def send_marker_with_time_stamp(self, marker_name):
        if self.connected:
            logger.info("iMotions (dummy) received the marker %s at %s", marker_name, self.time_stamp)

    def send_temperatures(self, temperature):
        if self.connected:
            logger.debug("iMotions (dummy) received temperature: %s", temperature)

    def send_ratings(self, rating):
        if self.connected:
            logger.debug("iMotions (dummy) received rating: %s", rating)

    def send_event_x_y(self, x, y):
        if self.connected:
            logger.debug("iMotions (dummy) received X, Y event: X=%s, Y=%s", x, y)

    def close(self):
        self.connected = False
        logger.info("iMotions (dummy) connection for event receiving closed.")
