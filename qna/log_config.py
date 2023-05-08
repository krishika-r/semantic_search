import logging
import logging.config

# Logging Config
# More on Logging Configuration
# https://docs.python.org/3/library/logging.config.html
# Setting up a config
LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "ARTHUR %(message)s"},
    },
    "root": {"level": "DEBUG"},
    "loggers": {},
}


def configure_logger(
    logger: logging = None,
    cfg: dict = None,
    log_file: str = None,
    console: bool = True,
    log_level: str = "DEBUG",
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------

    logger : logging
        Predefined logger object if present. If None a new logger object will be created from root.
    cfg : dict()
        Configuration of the logging to be implemented by default
    log_file : str
        Path to the log file for logs to be stored
    console : bool
        To include a console handler(logs printing in console)
    log_level : str
        One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
            default - `"DEBUG"`

    Returns
    -----------
    logger: logging.Logger
        A logger which contains the requested configuration
    """

    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            fh.setLevel(getattr(logging, log_level))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            sh.setLevel(getattr(logging, log_level))
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    return logger
