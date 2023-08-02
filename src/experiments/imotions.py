# work in progress

# TODO
# - how to close logger the right way
# - documentation, especially for the query structure
# - query structure / query types, esp. for event recieving
# - xml data in imotions
#    -> check if send_temperatures and send_ratings work
# - export data function, p. 34 onwards
# - Add option to connect that checks if imotions is avaiable or if you want to proceed without it by asking with input()
#    -> very handy for psychopy testing, where you don't want to have imotions connected all the time
# age and gender important for analysis in imotions? 


import socket
import logging
import time
from datetime import datetime

from .logger import setup_logger #, close_logger

logger = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)


class RemoteControliMotions():
    """ 
    Class to control iMotions remotely, based on the iMotions Remote Control API.
    The class has to be initated in a psychopy experiment, with the participant number and the study name.
    To import the class, add the folder containing the class to the python path in the psychopy preferences.

    The class does not work as a context manager, because the connection to iMotions has to be open during the whole experiment.

    Query structure:
        - R;2;TEST;STATUS\r\n ...

    Example
    -------

    """
    
    HOST = "localhost"
    PORT = 8087 # hardcoded in iMotions
    
    def __init__(self, study, participant):
        # Psychopy experiment info
        self.study = study # psychopy default is expName
        self.participant = participant # psychopy default is expInfo['participant']

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
            while not self.connected == 0:
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
        start_study_query = f"R;2;TEST;RUN;{self.study};{self.participant};Age={0} Gender={0}\r\n" 
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
    """"
    The events are only available for forwarding if the device is hooked up to 
    iMotions and iMotions has been configured to collect from the device.
    
    Single character that identifies the type of message. Two message types are 
    supported.
    E - Sensor Event
    M - Discrete Marker

    on psychopy level
    This class does not work as a context manager, because we need to send events every frame in the psychopy experiment.
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
            

if __name__ == "__main__":
    pass
