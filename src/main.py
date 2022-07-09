from ui.user_interface import MainWindow
from controller.controller import ClassificationController
from configs import config, logconfig
import argparse
import os
import logging

def main():
    if(not os.path.exists(logconfig.OUTPUT_LOG_FILE_DIR)): os.makedirs(logconfig.OUTPUT_LOG_FILE_DIR)

    logname = logconfig.OUTPUT_LOG_FILENAME
    logging.basicConfig(filename=os.path.join(logconfig.OUTPUT_LOG_FILE_DIR, logname), 
                        format = logconfig.LOG_FORMAT, datefmt=logconfig.LOG_DATE_FORMAT,
                        encoding='utf-8', 
                        level=logging.DEBUG)
    
    logging.info(logconfig.LOG_MAIN_START_OF_APPLICATION)

    parser = argparse.ArgumentParser()
    parser.add_argument(config.TRAIN_FLAG_SHORT, config.TRAIN_FLAG, help=config.TRAIN_FLAG_HELP, action=config.TRAIN_FLAG_ACTION)
    args = parser.parse_args()
    logging.info(logconfig.LOG_MAIN_FLAG_PARSING.format(t=args.train))
    
    controller = ClassificationController(args.train)
    MainWindow(controller).run()
    logging.info(logconfig.LOG_MAIN_END_OF_APPLICATION)

if __name__ == "__main__":
    main()
