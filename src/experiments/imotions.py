# work in progress

# TODO
# - close logger, maybe just put it into psychopy script 
# - add export data function, p. 34 onwards for experiment2
# - Add option to connect that checks if imotions is avaiable or if you want to proceed without it by asking with input()
#    -> could come in handy for psychopy testing, where you don't want to have imotions connected all the time
# - find out if age and gender are considered in the analysis in imotions
# find out what gender means in imotions, so far 0 was used..
# -> update function accordingly


import socket
import logging
import time
from datetime import datetime

from .logger import setup_logger #, close_logger

logger = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)


class RemoteControliMotions():
    """    
    This class provides an interface to control the iMotions software remotely based on the iMotions Remote Control API. 
    
    The class is designed to be integrated within a PsychoPy experiment, allowing for the initiation of studies, sending of commands, and receiving responses from the iMotions software.
    
    Attributes:
    -----------
    - HOST: The host address for the iMotions software, typically "localhost".
    - PORT: The port number used for communication, default is 8087 as hardcoded in iMotions.
    
    Methods:
    --------
    - __init__(self, study, participant): Initializes the class with study and participant details.
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
    imotions = RemoteControliMotions(study="StudyName", participant="ParticipantID")
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
    PORT = 8087 # hardcoded in iMotions
    
    def __init__(self, study, participant, age, gender):
        # Psychopy experiment info
        self.study = study # psychopy default is expName
        self.participant = participant # psychopy default is expInfo['participant']
        self.age = age # psychopy default is expInfo['age']
        self.gender = gender # psychopy default is expInfo['gender']

        # iMotions info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = None

    def _send_and_receive(self, query):
        """Helper function to send and receive data from iMotions."""
        self.sock.sendall(query.encode('utf-8'))
        response = self.sock.recv(1024).decode('utf-8')
        return response
    
    def _check_status(self):
        status_query = "R;2;TEST;STATUS\r\n"
        response = self._send_and_receive(status_query)
        # e.g. 1;RemoteControl;STATUS;;-1;TEST;1;;;;;0;;
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
            logger.info("iMotions is ready for remote control.")
        except socket.error as exc:
            logger.error("iMotions is not ready for remote control. Error connecting to server:\n%s", exc)
            raise Exception("iMotions is not ready for remote control. Error connecting to server:\n%s", exc) from exc

    def start_study(self):
        """
        Start study in iMotions.
        Note: iMotions always asks for age and gender. This could be reused for other variables.
        """
        # sent status request to iMotions and proceed if iMotions is ready
        if self._check_status() != 0:
            logger.error("iMotions is not ready the start the study.")
            raise Exception("iMotions is not ready the start the study.")
        start_study_query = f"R;2;TEST;RUN;{self.study};{self.participant};Age={self.age} Gender={self.gender}\r\n" 
        self._send_and_receive(start_study_query)
        logger.info("iMotions started the study %s for participant %s.", self.study, self.participant)

    def end_study(self):
        """
        End study in iMotions.
        """
        end_study_query = "R;1;;SLIDESHOWNEXT\r\n"
        self._send_and_receive(end_study_query)
        logger.info("iMotions ended the study %s for participant %s.", self.study, self.participant)

    def abort_study(self):
        """
        Abort study (Slide-show) in iMotions. Equivalent to pressing F11 in iMotions.
        """
        abort_study_query = "R;1;;SLIDESHOWCANCEL\r\n"
        self._send_and_receive(abort_study_query)
        logger.info("iMotions aborted the study %s for participant %s.", self.study, self.participant)

    def export_data(self):
        logger.info("iMotions exported the data for study %s for participant %s to %s.", self.study, self.participant, path)
        pass

    def close(self):
        try:
            self.sock.close()
            logger.info("iMotions connection for remote control closed.")
        except socket.error as exc:
            logger.error("iMotions connection for remote control could not be closed:\n%s", exc)
        finally:
            self.connected = None
        

class EventRecievingiMotions():
    """
    EventRecievingiMotions Class
    ----------------------------
    
    This class provides an interface to receive events from the iMotions software. 
    
    The class is designed to interface with external sensors or other third-party applications that can send data to the iMotions software. The received data is treated similarly to data collected from built-in sensors in iMotions, allowing for synchronization, visualization, storage, and export.
    
    Two message types are supported: 'E' for Sensor Event and 'M' for Discrete Marker.

    Attributes:
    -----------
    - HOST: The host address for the iMotions software, typically "localhost".
    - PORT: The port number used for communication, default is 8089 as hardcoded in iMotions.
    
    Methods:
    --------
    - __init__(self): Initializes the class and sets up the socket connection.
    - time_stamp(self): Returns the current timestamp.
    - connect(self): Establishes a connection to the iMotions software for event receiving.
    - _send_message(self, message): Sends a message to iMotions.
    - start_study(self): Sends a marker indicating the start of a study.
    - end_study(self): Sends a marker indicating the end of a study.
    - send_marker(self, marker_name, value): Sends a specific marker with a given value to iMotions.
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
    imotions_events.start_study()
    # ... capture events ...
    imotions_events.end_study()
    imotions_events.close()
    ```
    """
    
    HOST = "localhost"
    PORT = 8089 # hardcoded in iMotions
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._time_stamp = self.time_stamp # use self.time_stamp to get the current time stamp


    @property
    def time_stamp(self):
        self._time_stamp = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        return self._time_stamp

    def connect(self):
        try:
            self.sock.connect((self.HOST, self.PORT))
            logger.info("iMotions is ready for event recieving.")
        except socket.error as exc:
            logger.error("iMotions is not ready for event recieving. Error connecting to server:\n%s", exc)

    def _send_message(self, message):
        self.sock.sendall(message.encode('utf-8'))

    def start_study(self):
        start_study_marker = f"M;2;;;marker_start_study;{self.time_stamp};D;\r\n"
        self._send_message(start_study_marker)
        logger.info("iMotions received the marker for study start @ %s.", self.time_stamp[11:])

    def end_study(self):
        end_study_marker = f"M;2;;;marker_end_study;{self.time_stamp};D;\r\n"
        self._send_message(end_study_marker)
        logger.info("iMotions received the marker for study end @ %s.", self.time_stamp[11:])

    def send_marker(self, marker_name, value):
        imotions_marker = f"M;2;;;{marker_name};{value};D;\r\n"
        self._send_message(imotions_marker)
        logger.info("iMotions received the marker %s: %s.", marker_name, value)

    def send_marker_with_time_stamp(self, marker_name):
        imotions_marker = f"M;2;;;{marker_name};{self.time_stamp};D;\r\n"
        self._send_message(imotions_marker)
        logger.info("iMotions received the marker %s at %s.", marker_name, self.time_stamp[11:])

    def send_temperatures(self, temperature):
        """
        For sending the current temperature to iMotions every frame.
        See imotions_temperature.xml for the xml structure.
        """
        imotions_event = f"E;1;TemperatureCurve;1;;;;TemperatureCurve;{temperature}\r\n"
        self._send_message(imotions_event)

    def send_ratings(self, rating):
        """
        See imotions_rating.xml for the xml structure.
        """
        imotions_event = f"E;1;RatingCurve;1;;;;RatingCurve;{rating}\r\n"
        self._send_message(imotions_event)
        
    def send_event_x_y(self, x, y):
        """
        Shows up as two seperate data streams in iMotions, based on a generic xml class. 
        """
        values_imotions = f"E;1;GenericInput;1;0.0;;;GenericInput;{x};{y}\r\n"
        self._send_message(values_imotions)

    def close(self):
        try:
            self.sock.close()
            logger.info("iMotions connection for event recieving closed.")
        except socket.error as exc:
            logger.error("iMotions connection for event recieving could not be closed:\n%s", exc)
