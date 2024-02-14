import os  
import signal  
import sys 
from .langfuse_integration import langfuse_flush

def shutdwon_handler(*args):  
    """
    This function handles the shutdown process.
    
    It calls the langfuse_flush function to flush any pending changes,
    and then exits the program with a status code of 0.
    """
    langfuse_flush()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdwon_handler)  