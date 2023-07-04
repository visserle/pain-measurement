import socket
from datetime import datetime


class RemoteControliMotions():
    """ Class to control iMotions remotely, based on the iMotions Remote Control API.
    The class has to be initated in a psychopy experiment, with the participant number and the study name.
    To import the class, add the folder containing the class to the python path in the psychopy preferences.

    The class does not work as a context manager, because the connection to iMotions has to be open during the whole experiment.

    Query structure:
        - R;2;TEST;STATUS\r\n ... TODO
    """
    HOST = "localhost"
    PORT = 8087
    
    def __init__(self, participant, study_name, debug = False):
        # Psychopy experiment info
        self.participant = participant # in psychopy the default is expInfo['participant']
        self.study_name = study_name # in psychopy the default is expName

        # iMotions info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_imotions = None
        self.status_query = "R;2;TEST;STATUS\r\n"
        self.end_study_query = "R;1;;SLIDESHOWNEXT\r\n"
        self.start_study_query = f"R;2;TEST;RUN;{self.study_name};{self.participant};Age=0 Gender=1\r\n" # iMotions always asks for age and gender

        # Additional variables
        self.debug = debug

    def connect(self):
        self.sock.connect((self.HOST, self.PORT))
        # Check if iMotions is ready for remote control
        while not status_imotions == 0:
            # Send status request to iMotions
            self.sock.sendall(self.status_query.encode('utf-8'))
            # Receive the response from the server
            status_recv = self.sock.recv(1024).decode('utf-8')
            # for instance: 1;RemoteControl;STATUS;;-1;TEST;1;;;;;0;;
            status_imotions = int(status_recv.split(";")[-3])
        self.status_imotions = status_imotions
        if self.debug and self.status_imotions == 0:
            print("iMotions is ready for remote control.")
        return self.status_imotions

    def start_study(self):
        # Start experiment in iMotions
        self.sock.sendall(self.start_study_query.encode('utf-8'))
        # Get and print response
        if self.debug:
            start_study_recv = self.sock.recv(1024).decode('utf-8')
            print(f"iMotions started the study\n{start_study_recv}.")

    def end_study(self):
        # End study in iMotions
        self.sock.sendall(self.end_study_query.encode('utf-8'))
        # Get and print response
        if self.debug:
            end_study_recv = self.sock.recv(1024).decode('utf-8')
            print(f"iMotions ended the study\n{end_study_recv}.")

    def close(self):
        self.sock.close()
        self.status_imotions = None
        

class EventRecievingiMotions():
    
    HOST = "localhost"
    PORT = 8089
    
    def __init__(self, debug = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.debug = debug
        self._time_stamp = self.time_stamp # use self.time_stamp to get the current time stamp
        self.start_study_marker = f"M;2;;;start_study;{self.time_stamp};D;\r\n"
        self.end_study_marker = f"M;2;;;marker_end_routine;{self.time_stamp};D;\r\n"

    @property
    def time_stamp(self):
        self._time_stamp = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        return self._time_stamp # always use self.time_stamp to get the current time stamp
    
    def print_time_stamp(self):
        print(self.time_stamp)

    def connect(self):
        self.sock.connect((self.HOST, self.PORT))
        if self.debug:
            print("iMotions is ready for event recieving.")
    
    def send_marker_time_stamp(self, marker_name):
        self.sock.sendall(f"M;2;;;{marker_name};{self.time_stamp};D;\r\n".encode('utf-8'))
        if self.debug:
            print(f"iMotions recieved the marker\n{marker_name}")
			
    def send_marker_value(self, marker_name, value):
        self.sock.sendall(f"M;2;;;{marker_name};{value};D;\r\n".encode('utf-8'))
        if self.debug:
            print(f"iMotions recieved the marker\n{marker_name}")

    def start_study(self):
        self.sock.sendall(self.start_study_marker.encode('utf-8'))
        if self.debug:
            print(f"iMotions recieved the start study marker\n{self.start_study_marker}")

    def end_study(self):
        self.sock.sendall(self.end_study_marker.encode('utf-8'))
        if self.debug:
            print(f"iMotions recieved the end study marker\n{self.end_study_marker}")

    def close(self):
        self.sock.close()

if __name__ == "__main__":
    pass