import signal  
import sys 
from .langfuse_integration import langfuse_flush

def shutdown_handler(*args):  
    """
    This function handles the shutdown process.
    
    It calls the langfuse_flush function to flush any pending changes,
    and then exits the program with a status code of 0.
    """
    langfuse_flush()
    sys.exit(0)

# Register the shutdown_handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, shutdown_handler)

# Register the same shutdown_handler for SIGTERM
signal.signal(signal.SIGTERM, shutdown_handler)
