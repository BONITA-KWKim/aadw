import logging

def make_logger(name=None, level:str='info'):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  console = logging.StreamHandler()
  
  LEVEL = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
  }
  console.setLevel(LEVEL[level])
  console.setFormatter(formatter)

  logger.addHandler(console)

  return logger