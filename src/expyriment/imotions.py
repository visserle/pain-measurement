# work in progress

# TODO
# - add export data function, p. 34 onwards for experiment2
# - set to NoPrompt for data acquisition: TODO: really? maybe figure out best way to handle this
# - update doc strings
# - add UPD support
# - maybe switch to send two events at the same time? send_x_y

import itertools
import logging
import socket
import time
from datetime import datetime

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class iMotionsError(Exception):
    pass


class RemoteControliMotions:
    """
    This class provides an interface to control the iMotions software remotely based on the iMotions Remote Control API.

    The class is designed to be integrated within a PsychoPy experiment, allowing for the initiation of studies, sending of commands, and receiving responses from the iMotions software.

    Methods:
    --------
    - __init__(self, study, participant_info): Initializes the class with study and participant details.
    - _send_and_receive(self, query): Sends a query to iMotions and receives a response.
    - _check_status(self): Checks the status of the iMotions software.
    - connect(self): Establishes a connection to the iMotions software.
    - start_study(self): Initiates a study in iMotions.
    - end_study(self): Ends the current study in iMotions.
    - abort_study(self): Aborts the current study in iMotions.
    - export_data(self): Exports data from the iMotions software (implementation pending).
    - close(self): Closes the connection to the iMotions software.

    Example Usage:
    --------------
    ```python
    from src.experiments.imotions import RemoteControliMotions

    imotions = RemoteControliMotions(
        study="StudyName", participant_info={"participant": "P001", "age": 20, "gender": "Female"
    )
    # in a psychopy experiment use expName and expInfo['participant']
    imotions.connect()
    imotions.start_study()
    # run the experiment ...
    imotions.end_study()
    imotions.close()
    ```

    Notes:
    ------
    - Ensure that the iMotions software is running and the Remote Control API is enabled.
    - The class does not work as a context manager (because the connection to iMotions has to be open during the whole experiment),
      so the connection to iMotions should be managed manually using the connect() and close() methods.
    - The query structure for communication follows a specific format, e.g., "R;2;TEST;STATUS\\r\\n":
        - R: Represents a specific command or operation.
        - 2: Represent a version or type of the command.
        - TEST: Is a placeholder or specific command identifier.
        - STATUS: Is the actual command or operation to be performed.
        - \\r\\n: The end of the query (with only one backslash)
        -> See the iMotions Remote Control API documentation for more details and the different versions of the commands.
    """

    HOST = "localhost"
    PORT = 8087  # default port for iMotions remote control

    def __init__(self, study, participant_info: dict):
        # Experiment info
        self.study = study
        self.participant_info = participant_info

        # iMotions info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # longer timeout to have time to react to iMotions prompts (if enabled via start_study mode)
        self.sock.settimeout(30.0)
        self.connected = None

    def _send_and_receive(self, query):
        """Helper function to send and receive data from iMotions."""
        self.sock.sendall(query.encode("utf-8"))
        response = self.sock.recv(1024).decode("utf-8").strip()
        return response

    def _check_status(self):
        """
        Helper function to check the status of iMotions.
        Returns 0 if iMotions is ready for remote control.
        """
        status_query = "R;2;;STATUS\r\n"
        response = self._send_and_receive(status_query)
        # e.g. 1;RemoteControl;STATUS;;-1;;1;;;;;0;;
        return int(response.split(";")[-3])

    def connect(self):
        """
        Check if iMotions is ready for remote control.
        """
        try:
            self.sock.connect((self.HOST, self.PORT))
            while self.connected != 0:
                self.connected = self._check_status()
                time.sleep(0.1)
            logger.debug("Ready for remote control.")
        except socket.error as exc:
            logger.error("Not ready for remote control. Error connecting to server:\n%s", exc)
            raise iMotionsError(
                "Not ready for remote control. Error connecting to server:\n%s", exc
            ) from exc

    def start_study(self, mode="NormalPrompt"):
        """
        Start study in iMotions with participant details.

        Notes
        -----
        There are three prompt handling commands in v3:
        - NormalPrompt: Default behavior where the operator is prompted to confirm continuing when certain conditions are detected e.g. an expected sensor is not active.
        - NoPromptIgnoreWarnings: The operator is not prompted on warnings, it is assumed that the continue option is desired.
        - NoPrompt: The operator is not prompted on warnings, they are treated as errors, and the study will not be run.
        """
        # sent status request to iMotions and proceed if iMotions is ready
        if self._check_status() != 0:
            logger.error("Not ready to start study.")
            raise iMotionsError("Not ready to start study.")
        start_study_query = f"R;3;;RUN;{self.study};{self.participant_info['id']};Age={self.age} Gender={self.gender};{mode}\r\n"
        response = self._send_and_receive(start_study_query)
        # e.g. "13;RemoteControl;RUN;;-1;;1;"
        logger.info(
            "Started recording participant %s (%s)", self.participant_info["id"], self.study
        )
        response_msg = response.split(";")[-1]
        if len(response_msg) > 0:  # if there is a response message, something went wrong
            logger.error("iMotions error code: %s", response_msg)
            raise iMotionsError("iMotions error code: %s", response_msg)

    def end_study(self):
        """
        End study in iMotions.
        """
        end_study_query = "R;1;;SLIDESHOWNEXT\r\n"
        self._send_and_receive(end_study_query)
        logger.info(
            "Stopped recording participant %s (%s)", self.participant_info["id"], self.study
        )

    def abort_study(self):
        """
        Abort study (Slide-show) in iMotions. Equivalent to pressing F11 in iMotions.
        """
        abort_study_query = "R;1;;SLIDESHOWCANCEL\r\n"
        self._send_and_receive(abort_study_query)
        logger.info(
            "Aborted recording participant %s (%s)", self.participant_info["id"], self.study
        )

    def export_data(self, path):
        logger.debug("Exported data for participant %s to %s", self.participant_info["id"], path)
        pass

    def close(self):
        try:
            self.sock.close()
            logger.debug("Closed remote control connection.")
        except socket.error as exc:
            logger.error("Error closing remote control connection:\n%s", exc)
        finally:
            self.connected = None


class EventRecievingiMotions:
    """
    This class provides an interface to receive events from the iMotions software.

    The class is designed to interface with external sensors or other third-party applications that can send data to the iMotions software. The received data is treated similarly to data collected from built-in sensors in iMotions, allowing for synchronization, visualization, storage, and export.

    Two message types are supported: 'E' for Sensor Event and 'M' for Discrete Marker.

    Methods:
    --------
    - __init__(self): Initializes the class and sets up the socket connection.
    - time_stamp(self): Returns the current timestamp.
    - connect(self): Establishes a connection to the iMotions software for event receiving.
    - _send_message(self, message): Sends a message to iMotions.
    - send_marker(self, marker_name, value): Sends a specific marker with a given value to iMotions.
    - send_stimulus_markers(self, seed): Sends a start and end stimulus marker for a given seed value of a stimulus function.
    - send_prep_markers(self): Sends a start and end marker for the preparation phase.
    - send_marker_with_time_stamp(self, marker_name): Sends a marker with the current timestamp.
    - send_temperatures(self, temperature): Sends the current temperature reading to iMotions.
    - send_ratings(self, rating): Sends a rating value to iMotions.
    - send_event_x_y(self, x, y): Sends x and y values as separate data streams to iMotions.
    - close(self): Closes the connection to the iMotions software.

    Notes:
    ------
    - The class interfaces with iMotions using either a UDP or TCP network connection. For this class TCP is used.
    - Event reception in iMotions is enabled using the Global Settings dialog within the API tab.
    - The incoming data packet must conform to a specific specification. When a packet is received, it's processed and checked against the registry of configured event sources.
    - iMotions can receive data from many event sources, and each event source can support multiple sample types. An additional event source definition file (XML text file) is used to describe the samples that can be received from a source.
    - This class does not work as a context manager, because we need to send events every frame in the psychopy experiment. The connection to iMotions should be managed manually using the connect() and close() methods.


    Example Usage:
    --------------
    ```python
    imotions_events = EventRecievingiMotions()
    imotions_events.connect()
    send_stimulus_markers(seed=9)  # sends a start stimulus marker for seed 9
    send_stimulus_markers(seed=9)  # can be called again to send an end stimulus marker for seed 9
    imotions_events.close()
    ```
    """

    HOST = "localhost"
    PORT = 8089  # default port for iMotions event recieving

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._time_stamp = self.time_stamp  # use self.time_stamp to get the current time stamp
        self.seed_cycles = {}  # class variable to keep track of seed cycles (start and end stimulus markers)
        self.prep_cycle = itertools.cycle(
            ["M;2;;;thermode_ramp_on;;D;\r\n", "M;2;;;thermode_ramp_off;;D;\r\n"]
        )

    @property
    def time_stamp(self):
        # Use self.time_stamp[11:] to get the time stamp without the date.
        self._time_stamp = datetime.utcnow().isoformat(sep=" ", timespec="milliseconds")
        return self._time_stamp

    def connect(self):
        try:
            self.sock.connect((self.HOST, self.PORT))
            logger.debug("Ready for event recieving.")
        except socket.error as exc:
            logger.error("Not ready for event recieving. Error connecting to server:\n%s", exc)
            raise iMotionsError(
                f"Not ready for event recieving. Error connecting to server:\n{exc}"
            ) from exc

    def _send_message(self, message):
        self.sock.sendall(message.encode("utf-8"))

    def send_marker(self, marker_name, value):
        imotions_marker = f"M;2;;;{marker_name};{value};D;\r\n"
        self._send_message(imotions_marker)
        logger.debug("Received marker %s: %s.", marker_name, value)

    def send_stimulus_markers(self, seed):
        """
        Alternates between generating 'start of seed' and 'end of seed' stimulus markers for each unique seed value.

        This function creates and maintains a separate cycling state for each seed, ensuring that each call for a
        particular seed alternates between 'start' and 'end' markers. A new cycle is initialized for each new seed.
        """
        if seed not in self.seed_cycles:  # only create a new cycle if it doesn't exist yet
            self.seed_cycles[seed] = itertools.cycle(
                [f"M;2;;;heat_stimulus;{seed};S;\r\n", f"M;2;;;heat_stimulus;{seed};E;\r\n"]
            )
        self._send_message(next(self.seed_cycles[seed]))
        logger.debug("Received stimulus marker for seed %s.", seed)

    def send_prep_markers(self):
        """Marker to indicate the start and end the ramp on/ramp off of the heat stimulus.
        Directly before and after the prep marker we are at baseline temperature."""
        self._send_message(next(self.prep_cycle))
        logger.debug("Received marker for thermode ramp on/off.")

    def send_temperatures(self, temperature, debug=False):
        """
        For sending the current temperature to iMotions every frame.
        See imotions_temperature.xml for the xml structure.
        """
        imotions_event = f"E;1;TemperatureCurve;1;;;;TemperatureCurve;{temperature:.2f}\r\n"
        self._send_message(imotions_event)
        if debug:
            logger.debug("Received temperature: %s.", temperature)

    def send_ratings(self, rating, debug=False):
        """
        See imotions_rating.xml for the xml structure.
        """
        imotions_event = f"E;1;RatingCurve;1;;;;RatingCurve;{rating:.2f}\r\n"
        self._send_message(imotions_event)
        if debug:
            logger.debug("Received rating: %s.", rating)

    def send_data(self, temperature, rating, debug=False):
        """
        Send temperature and rating data to iMotions in one go. 
        """
        imotions_data = f"E;1;CustomCurves;1;;;;CustomCurves;{temperature};{rating}\r\n"
        self.send_messages(imotions_data)
        if debug:
            logger.debug("Received temperature: %s, rating: %s.", temperature, rating)

    def send_event_x_y(self, x, y):
        """
        Shows up as two seperate data streams in iMotions, based on a generic xml class.
        """
        values_imotions = f"E;1;GenericInput;1;0.0;;;GenericInput;{x};{y}\r\n"
        self._send_message(values_imotions)

    def close(self):
        try:
            self.sock.close()
            logger.info("Closed event recieving connection.")
        except socket.error as exc:
            logger.error("Error closing event recieving connection:\n%s", exc)
