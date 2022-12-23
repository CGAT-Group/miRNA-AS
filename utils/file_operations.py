import os
import time
import logging
logger = logging.getLogger(__name__)

#check regulary every interval_time seconds whether file is sucessfully written else print error but maximal try for max_time seconds
def check_whether_exists(file, interval_time, max_time, error_file=None):
    elapsed_time = 0
    while not os.path.exists(file):
        time.sleep(interval_time)
        elapsed_time += interval_time
        if (error_file != None):
            if os.path.exists(error_file) and os.path.getsize(error_file) > 0:
                with open(error_file) as f:
                    logger.error(f.read())
                exit()
        if (elapsed_time > max_time):
            logger.error(f'Waited longer than {max_time}.')
            exit()