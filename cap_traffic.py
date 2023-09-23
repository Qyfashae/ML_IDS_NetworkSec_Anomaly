#Coded by: QyFashae

import pyshark # Importing pyshark"wireshark"
import datetime
import pandas as pd

# Specify the duration for the capture.
capture_time = 1085 

# Specify the name of the file to output.
start = datetime.datetime.now() # Starting the time of the network data to capture
end = start+datetime.datetimedelta(seconds=capture_time) # The end time of the network data capture
file_name = "net_traffic"+str(start).replace(" ", "T")+"to"+str(end).replace(" ", "T")+".pcap"

# Capture network traffic
cap = pyshark.LiveCapture(ouput_filename=output_filename)
cap.sniff(timeout=capture_time)

print("Open the @net_traffic pcap file for manuall investigation.")
exit(0)
