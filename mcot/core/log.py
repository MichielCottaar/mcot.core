from loguru import logger
import os
import sys


SUBMITTED = os.getenv('FSLSUBALREADYRUN', default='') == 'true'


def get_level():
    """
    Returns the requested log level

    0: no logging
    1: warning+
    2: info+
    3: debug+
    4: all+
    """
    levels = (
        None,
        "WARNING",
        "INFO",
        "DEBUG",
        0
    )
    if 'MCLOG' in os.environ:
        idx = int(os.environ['MCLOG'])
    else:
        idx = 3 if SUBMITTED else 2
    if idx > 4:
        idx = -1
    return levels[idx]


LOGSET = False


def setup_log(filename=None, replace=True):
    """
    Sets up the logger to log to the given filename

    Sets up logging to:

    - If .log directory exists:

      - ~/.log/python.info (INFO and higher)
      - ~/.log/python.debug (DEBUG and higher)

    - stderr (minimum level is set by MCLOG; default values are DEBUG if submitted, INFO otherwise)

    - `filename` if provided

    :param filename: filename used in the logging (default: only set up logging to stdout/stderr and ~/.log directory)
    :param replace: if replace is True, replace the existing handlers
    """
    global LOGSET

    if replace:
        logger.remove(handler_id=None)

    if get_level() is None:
        return

    if filename is not None:
        logger.add(filename, level="DEBUG")

    if not LOGSET:
        logger.add(sys.stderr, level=get_level())

        log_directory = os.path.expanduser('~/.log')
        if os.path.isdir(log_directory):
            try:
                logger.add(os.path.join(log_directory, 'python.info'), rotation='1:00', level="INFO")
            except OSError:
                logger.exception("Failed to create .log/python.info file")
            try:
                logger.add(os.path.join(log_directory, 'python.debug'), rotation='1:00', level="DEBUG")
            except OSError:
                logger.exception("Failed to create .log/python.debug file")
        else:
            logger.info('Log directory %s not found; Skipping setup of global logging', log_directory)

        LOGSET = True
        logger.info('Initial logging setup to %s complete', filename)
    else:
        logger.info('Additional logging to %s setup', filename)


def log_function(verbose=True, include_parameters=False, include_result=False):
    """Log the time spent in a function.

    When verbose: log both entering and leaving of function, otherwise only when leaving function.
    """
    def towrap(func):
        name = getattr(func, '__module___', repr(func))
        logger = getLogger(name)
        @wraps(func)
        def wrapped(*args, **kwargs):
            if verbose:
                if include_parameters:
                    logger.info('Calling %s(%s, %s)' % (func.__name__, str(args)[1:-1], ', '.join(['%s = %s' % item for item in kwargs.items()])))
                else:
                    logger.info('Calling %s' % func)
            time_start = time.time()
            result = func(*args, **kwargs)
            total_time = time.time() - time_start
            msg = 'Done %s%s%s in %.3f sec' % (("%s = " % result if include_result else ""), func.__name__, ("(%s, %s)" % (str(args)[1:-1], ', '.join(['%s = %s' % item for item in kwargs.items()])) if include_parameters else ""), total_time)
            logger.info(msg)
            return result
        return wrapped
    return towrap
