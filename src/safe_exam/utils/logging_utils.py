import logging


def configure_logging(level=logging.INFO):
    """
    Configure the logging for the application.
    :param level: the logging level
    :return: None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
