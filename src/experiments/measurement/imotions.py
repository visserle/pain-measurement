import itertools
import logging
import socket
import time
from collections import namedtuple
from pathlib import Path

from src.experiments.measurement.rate_limiter import RateLimiter

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class iMotionsError(Exception):
    pass


class DummySocket:
    def __init__(self, *args, **kwargs):
        self.response = None

    def connect(self, *args, **kwargs):
        pass

    def settimeout(self, *args, **kwargs):
        pass

    def sendall(self, command):
        command = command.decode()
        if "STATUS" in command:
            self.response = "1;RemoteControl;STATUS;;-1;;1;;;;;0;;"
        elif "RUN" in command:
            self.response = "13;RemoteControl;RUN;;-1;;1;"
        elif "SLIDESHOWNEXT" in command:
            self.response = ""

    def recv(self, *args, **kwargs):
        return self.response.encode()

    def close(self, *args, **kwargs):
        pass


class RemoteControliMotions:
    """
    This class provides an interface to control the iMotions software remotely.

    The class is designed to be integrated within an experiment, allowing for the
    initiation of studies, sending of commands, and receiving responses from
    the iMotions software.

    Methods:
    --------
    - __init__(self, study, participant_info, dummy=False): Initializes the class with
    study and participant details.
    - connect(self): Establishes a connection to the iMotions software.
    - start_study(self): Initiates a study in iMotions.
    - end_study(self): Ends the current study in iMotions.
    - abort_study(self): Aborts the current study in iMotions.
    - export_data(self): Exports data from the iMotions software.
    - close(self): Closes the connection to the iMotions software.

    Example Usage:
    --------------
    ```python
    from src.experyment.imotions import RemoteControliMotions

    imotions_control = RemoteControliMotions(
        study="dummy_study",
        participant_info={"id": "P001", "age": 20, "gender": "Female"},
        dummy=True,
    )
    imotions_control.connect()
    imotions_control.start_study()
    # run the experiment ...
    imotions_control.end_study()
    imotions_control.close()
    ```

    Notes:
    ------
    - Ensure that the iMotions software is running with the Remote Control API enabled.
    - The query structure for communication follows a specific format,
      e.g., "R;2;TEST;STATUS\\r\\n", where the query parts are:
        - R: Represents a specific command or operation.
        - 2: Represent a version or type of the command. Different queries need
          different versions.
        - TEST: Is a placeholder or specific command identifier. Not used here.
        - STATUS: Is the actual command or operation to be performed. Always at index 3.
        - \\r\\n: The end of the query (with only one backslash)
    -> See the iMotions Remote Control API documentation for more details and
       the different versions of the commands.
    """

    HOST = "localhost"
    PORT = 8087  # default port for iMotions remote control

    def __init__(
        self,
        study: str,
        participant_info: dict,
        dummy: bool = False,
    ):
        # Experiment info
        self.study = study
        self.participant_info = participant_info
        self.dummy = dummy
        if self.dummy:
            logger.debug("Running in dummy mode.")

        # iMotions info
        self.sock = (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if not dummy
            else DummySocket()
        )
        # longer timeout to have time to react to iMotions prompts
        # (if enabled via start_study mode)
        self.sock.settimeout(30.0)
        self.connected = None

    def _send_and_receive(self, query):
        """Helper function to send and receive data from iMotions."""
        self.sock.sendall(query.encode("utf-8"))
        response = self.sock.recv(1024).decode("utf-8").strip()
        return response

    def _validate_response(self, response):
        """Helper function to validate the response from iMotions."""
        if response:
            msg = response.split(";")[-1]
            # if there is a response message, something went wrong for most commands
            if msg and not Path(msg).exists():  # ignore file paths
                logger.error(
                    "iMotions error code for command %s: %s.",
                    response.split(";")[2],
                    msg,
                )
                return False
        return True

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
            msg = f"Not ready for remote control. Error connecting to server:\n{exc}"
            logger.error(msg)
            raise iMotionsError(msg) from exc

    def start_study(self, mode: str = "NormalPrompt"):
        """
        Start study in iMotions with participant details.

        Notes
        -----
        There are three prompt handling commands in v3:
        - NormalPrompt: Default behavior where the operator is prompted to confirm
          continuing when certain conditions are detected e.g. an expected sensor is
          not active.
        - NoPromptIgnoreWarnings: The operator is not prompted on warnings, it is
          assumed that the continue option is desired.
        - NoPrompt: The operator is not prompted on warnings, they are treated as
          errors, and the study will not be run.
        """
        # sent status request to iMotions and proceed if iMotions is ready
        if self._check_status() != 0:
            logger.error("Not ready to start study.")
            raise iMotionsError("Not ready to start study.")
        gender = "Female" if self.participant_info["gender"] == "f" else "Male"
        start_study_query = (
            f"R;3;;RUN;{self.study};{self.participant_info['id']};"
            f"Age={self.participant_info['age']} Gender={gender};{mode}\r\n"
        )
        response = self._send_and_receive(start_study_query)
        # e.g. "13;RemoteControl;RUN;;-1;;1;"
        if not self._validate_response(response):
            raise iMotionsError("Error starting study.")
        logger.info(
            "Started recording participant %s (%s).",
            self.participant_info["id"],
            self.study,
        )

    def end_study(self):
        """
        End study in iMotions.
        """
        end_study_query = "R;1;;SLIDESHOWNEXT\r\n"
        self._send_and_receive(end_study_query)
        logger.info(
            "Stopped recording participant %s (%s).",
            self.participant_info["id"],
            self.study,
        )

    def abort_study(self):
        """
        Abort study (Slide-show) in iMotions. Equivalent to pressing F11 in iMotions.
        """
        abort_study_query = "R;1;;SLIDESHOWCANCEL\r\n"
        self._send_and_receive(abort_study_query)
        logger.info(
            "Aborted recording participant %s (%s).",
            self.participant_info["id"],
            self.study,
        )

    def export_data(self, dir_path: Path):
        """
        Export data from iMotions to a given directory path.

        Note that this function will throw an error if iMotions is still busy with a
        previous command.
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Export data for participant %s to %s.",
            self.participant_info["id"],
            dir_path,
        )
        export_query = (
            f"R;1;;EXPORTSENSORDATA;;{self.study};"
            f"{self.participant_info['id']};{dir_path.resolve()};;\r\n"
        )
        response = self._send_and_receive(export_query)
        self._validate_response(response)

    def close(self):
        try:
            self.sock.close()
            logger.debug("Closed remote control connection.")
        except socket.error as exc:
            logger.error("Error closing remote control connection:\n%s", exc)
        finally:
            self.connected = None


DataPoint = namedtuple("DataPoint", ["timestamp", "temperature", "rating"])


class EventRecievingiMotions:
    """
    This class provides an interface to send discrete markers / continuous event data to
    the iMotions software.

    Methods:
    --------
    - __init__(self, sample_rate, dummy=False): Initializes the class with the sample
      rate for the rate limiter and a dummy mode.
    - connect(self): Establishes a connection to the iMotions software for event
      receiving.
    - send_marker(self, marker_name, value): Sends a specific marker with a given value
      to iMotions.
    - send_prep_markers(self): Sends a start and end marker for the preparation phase of
      the pain stimulus (ramp on and off).
    - send_stimulus_markers(self, seed): Sends a start/end marker for a given seed value
      of a stimulus function.
    - send_data_rate_limited(self, timestamp, temperature, rating, debug=False): Sends
      temperature and rating data from the pain-measurement experiment to iMotions.
    - close(self): Closes the connection to the iMotions software.

    Notes:
    ------
    - The message types are: 'E' for Sensor Event and 'M' for Discrete Marker.
    - iMotions supports both TCP and UDP network connections for event receiving.
      In this class, TCP is used (more reliable and ordered data transfer compared
      to UDP, but slower).
    - iMotions can receive data from many event sources, and each event source can
      support multiple sample types.
    - An additional event source definition file (XML text file) is used to describe the
      samples that can be received from a source.

    Example Usage:
    --------------
    ```python
    imotions_events = EventRecievingiMotions(sample_rate, dummy=True)
    imotions_events.connect()
    # Send a start stimulus marker for seed 9
    imotions_events.send_stimulus_markers(seed=9)
    # Call again to send an end stimulus marker for the same seed
    imotions_events.send_stimulus_markers(seed=9)
    imotions_events.close()
    ```
    """

    HOST = "localhost"
    PORT = 8089  # default port for iMotions event recieving

    def __init__(
        self,
        sample_rate: int,
        dummy: bool = False,
    ):
        """
        Initialize the class with the sample rate for the rate limiter and optional
        dummy mode.
        """
        self.sock = (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if not dummy
            else DummySocket()
        )
        # Class variables to keep track of the cycling states for markers
        self.seed_cycles = {}
        self.prep_cycle = itertools.cycle(
            [
                "M;2;;;thermode_ramp_on/off;;S;\r\n",
                "M;2;;;thermode_ramp_on/off;;E;\r\n",
            ]
        )
        self.rate_limiter = RateLimiter(sample_rate, use_intervals=True)
        self.dummy = dummy

        self.data_points = []

    def connect(self) -> None:
        try:
            self.sock.connect((self.HOST, self.PORT))
            logger.debug("Ready for event recieving.")
        except socket.error as exc:
            logger.error(
                "Not ready for event recieving. Error connecting to server:\n%s", exc
            )
            raise iMotionsError(
                f"Not ready for event recieving. Error connecting to server:\n{exc}"
            ) from exc

    def _send_message(self, message):
        self.sock.sendall(message.encode("utf-8"))

    def send_marker(
        self,
        marker_name: str,
        value: str | int | float,
    ) -> None:
        imotions_marker = f"M;2;;;{marker_name};{value};D;\r\n"
        self._send_message(imotions_marker)
        logger.debug("Received marker %s: %s.", marker_name, value)

    def send_prep_markers(self) -> None:
        """
        Sends a start and end marker for the preparation phase of the pain stimulus
        (ramp on and off).

        Uses a cycling state to alternate between the two markers.
        """
        self._send_message(next(self.prep_cycle))
        logger.debug("Received marker for thermode ramp on/off.")

    def send_stimulus_markers(self, seed: int) -> None:
        """
        Sends a start (S) and end (E) marker for a given seed value of a stimulus
        function.

        This function creates and maintains a separate cycling state for each seed,
        ensuring that each call for a particular seed alternates between start and end
        markers. A new cycle is initialized for each new seed.
        """
        if (
            seed not in self.seed_cycles
        ):  # only create a new cycle if it doesn't exist yet
            self.seed_cycles[seed] = itertools.cycle(
                [
                    f"M;2;;;pain_stimulus;{seed};S;\r\n",
                    f"M;2;;;pain_stimulus;{seed};E;\r\n",
                ]
            )
        self._send_message(next(self.seed_cycles[seed]))
        logger.debug("Received stimulus marker for seed %s.", seed)

    def send_data_rate_limited(
        self,
        timestamp: int,
        temperature: float,
        rating: float,
        debug: bool = False,
    ) -> None:
        """
        Send temperature and rating data to iMotions at once.

        This function uses an interval-based rate limiter to ensure that the data is
        sent at a specific sampling rate.

        See imotions.xml for the xml structure.
        """
        if self.rate_limiter.is_allowed(timestamp):
            imotions_data = (
                f"E;1;CustomCurves;1;;;;CustomCurves;{temperature};{rating}\r\n"
            )
            self._send_message(imotions_data)

            data_point = DataPoint(timestamp, temperature, rating)
            self.data_points.append(data_point)
            if debug:
                logger.debug(
                    "Time: %s. Received temperature: %s, rating: %s.",
                    timestamp,
                    temperature,
                    rating,
                )

    def clear_data_points(self) -> None:
        """
        Clear the data points stored in the class.
        """
        self.data_points = []

    def close(self) -> None:
        """
        Close the connection to the iMotions software.
        """
        try:
            self.sock.close()
            logger.debug("Closed event recieving connection.")
        except socket.error as exc:
            logger.error("Error closing event recieving connection:\n%s", exc)


def main():
    import logging

    # Set up very basic logging
    logging.basicConfig(level=logging.DEBUG)

    # Remote control example
    imotions = RemoteControliMotions(
        study="pain-measurement",
        participant_info={"id": "0", "age": 0, "gender": "Female"},
        dummy=False,
    )

    imotions.connect()
    imotions.export_data(Path("data/imotions"))
    # imotions.connect()
    # imotions.start_study()
    # # run the experiment ...
    # imotions.end_study()
    # imotions.close()

    # # Event recieving example
    # imotions_events = EventRecievingiMotions(sample_rate=10, dummy=True)
    # imotions_events.connect()
    # # Send a start stimulus marker for seed 9
    # imotions_events.send_stimulus_markers(seed=9)
    # # Call again to send an end stimulus marker for the same seed
    # imotions_events.send_stimulus_markers(seed=9)
    # imotions_events.close()


if __name__ == "__main__":
    main()
