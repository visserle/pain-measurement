# work in progress

# TODO
# - documentation, especially for the query structure
# - logging
# - query structure / query types
# - xml data in imotions
# - new names for data streams
# - Add option to connect that checks if imotions is avaiable or if you wwant to proceed without it by asking with input()
#    -> very handy for psychopy testing, where you don't want to have imotions connected all the time
# - remote contol does not work as a context manager, but maybe event recieving does

import socket
import logging
from datetime import datetime


class RemoteControliMotions():
    """ 
    Class to control iMotions remotely, based on the iMotions Remote Control API.
    The class has to be initated in a psychopy experiment, with the participant number and the study name.
    To import the class, add the folder containing the class to the python path in the psychopy preferences.

    The class does not work as a context manager, because the connection to iMotions has to be open during the whole experiment.

    Query structure:
        - R;2;TEST;STATUS\r\n ...
    """
    
    HOST = "localhost"
    PORT = 8087 # hardcoded in iMotions
    
    def __init__(self, study, participant):
        # Psychopy experiment info
        self.study = study # psychopy default is expName
        self.participant = participant # psychopy default is expInfo['participant']

        # iMotions info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_connected = False
        self.status_query = "R;2;TEST;STATUS\r\n"

        # Additional variables
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) # debug, info, warning, error, critical
        if not self.logger.handlers:  # Check if the logger already has a handler for use in e.g. jupyter notebooks
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(handler)

    def connect(self):
        """
        Check if iMotions is ready for remote control.
        """
        try:
            self.sock.connect((self.HOST, self.PORT))
            while not self.status_imotions == 0:
                # Send status request to iMotions
                self.sock.sendall(self.status_query.encode('utf-8'))
                # Receive the response from the server
                status_recv = self.sock.recv(1024).decode('utf-8')
                # e.g. 1;RemoteControl;STATUS;;-1;TEST;1;;;;;0;;
                self.connected = int(status_recv.split(";")[-3])
            self.logger.info("iMotions is ready for remote control.")
        except socket.error as exc:
            self.logger.error(f"Error connecting to server: {exc}")

    def _send_and_receive(self, query):
        """Helper function to send and receive data from iMotions."""
        self.sock.sendall(query.encode('utf-8'))
        response = self.sock.recv(1024).decode('utf-8')
        return response

    def start_study(self):
        """
        Start study in iMotions.
        Note: iMotions always asks for age and gender, this could be reused for other variables.
        """
        start_study_query = f"R;2;TEST;RUN;{self.study};{self.participant};Age=0 Gender=0\r\n" 
        # sent status request to iMotions and proceed if iMotions is ready
        response = self._send_and_receive(self.status_query)
        if int(response.split(";")[-3]) != 0:
            self.logger.error("iMotions is not ready the start the study.")
            raise Exception("iMotions is not ready the start the study.")
        # start study
        self._send_and_receive(start_study_query)
        self.logger.info(f"iMotions started the study {self.study}.")

    def end_study(self):
        """
        End study in iMotions.
        """
        end_study_query = "R;1;;SLIDESHOWNEXT\r\n"
        self._send_and_receive(end_study_query)
        self.logger.info(f"iMotions ended the study {self.study}.")

    def abort_study(self):
        """
        Abort study (Slide-show) in iMotions. Equivalent to pressing F11 in iMotions.
        """
        abort_study_query = "R;1;;SLIDESHOWCANCEL\r\n"
        self._send_and_receive(abort_study_query)
        self.logger.info(f"iMotions aborted the study {self.study}.")

    def export_data(self):
        # TODO: implement
        # p. 34 onwards
        pass

    def close(self):
        self.sock.close()
        self.status_imotions = None
        self.logger.info("iMotions connection for remote control closed.")
        

class EventRecievingiMotions():
    """"
    The events are only available for forwarding if the device is hooked up to 
    iMotions and iMotions has been configured to collect from the device.
    
    Single character that identifies the type of message. Two message types are 
    supported.
    E - Sensor Event
    M - Discrete Marker

    on psychopy level
    """
    
    HOST = "localhost"
    PORT = 8089 # hardcoded in iMotions
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._time_stamp = self.time_stamp # use self.time_stamp to get the current time stamp
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) # debug, info, warning, error, critical
        if not self.logger.handlers:  # Check if the logger already has a handler for use in e.g. jupyter notebooks
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(handler)


    @property
    def time_stamp(self):
        self._time_stamp = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        return self._time_stamp

    def connect(self):
        try:
            self.sock.connect((self.HOST, self.PORT))
            self.logger.info("iMotions is ready for event recieving.")
        except socket.error as e:
            self.logger.error(f"Error connecting to server: {e}")

    def _send_message(self, message):
        self.sock.sendall(message.encode('utf-8'))

    def send_time_stamp(self, marker_name):
        message = f"M;2;;;{marker_name};{self.time_stamp};D;\r\n"
        self._send_message(message)
        self.logger.info(f"iMotions received the marker\n{marker_name} at {self.time_stamp[11:]}")

    def send_value_marker(self, marker_name, value):
        message = f"M;2;;;{marker_name};{value};D;\r\n"
        self._send_message(message)
        self.logger.info(f"iMotions received the marker\n{marker_name}")

    def send_value_event(self, marker_name):
        pass

    def start_study(self):
        start_study_marker = f"M;2;;;marker_start_study;{self.time_stamp};D;\r\n"
        self._send_message(start_study_marker)
        self.logger.info(f"iMotions received the start study marker\n{start_study_marker}") 

    def end_study(self):
        end_study_marker = f"M;2;;;marker_end_study;{self.time_stamp};D;\r\n"
        self._send_message(end_study_marker)
        self.logger.info(f"iMotions received the end study marker\n{end_study_marker}")

    def send_temperatures(self, temperature):
        pass

    def send_ratings(self, ratings):
        pass

    def event_x_y(self, x, y):
        """
        one xml class (?), shows up as two seperate data streams in imotions
        """
        values_imotions = f"E;1;GenericInput;1;0.0;;;GenericInput;{x};{y}\r\n"
        self._send_message(values_imotions)

    def event_x_y_2(self, x, y):
        values_imotions = f"E;1;GenericInput2;1;0.0;;;GenericInput2;{x};{y}\r\n"
        self._send_message(values_imotions)

    def close(self):
        try:
            self.sock.close()
            self.logger.info("iMotions connection for event recieving closed.")
        except socket.error as e:
            self.logger.error(f"Error closing connection to server: {e}")

if __name__ == "__main__":
    pass