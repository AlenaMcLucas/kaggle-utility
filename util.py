

########## LOGGING ##########

import logging

# # Create and configure logger 
logging.basicConfig(filename="logs/pipeline.log",
	level=logging.DEBUG, format='%(asctime)s %(message)s', filemode='a')
  

# Configure messages
def log(message, module, level):
	if level == "info":
		logging.info(module + "\n" + message)
	elif level == "debug":
		logging.debug(module + "\n" + "Debugging: " + message)
	else:
		logging.info("level has not been coded")

####################



