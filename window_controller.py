"""
The purpose of this file is to allow for it to be called upon to control a window (ie: given a execution call from a file such as *.py or main.py itself). 
This will be the entrypoint in which will be used to determnistically choose which application should be opened. 

For now we will only accept the application of apple.com to be opened. 
"""

import subprocess

def open_apple():
    subprocess.Popen([
        "cmd", "/c", "start", "chrome",
        "--new-window",
        "--window-position=0,0",
        "--start-fullscreen",
        "https://apple.com"
    ])