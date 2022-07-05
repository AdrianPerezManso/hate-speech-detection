from ui.interface import print_interface
from ui.user_interface import MainWindow, AuthenticationWindow, TrainingWindow, DialogWindow
from controller.controller import ClassificationController
from configs import config
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(config.TRAIN_FLAG_SHORT, config.TRAIN_FLAG, help=config.TRAIN_FLAG_HELP, action=config.TRAIN_FLAG_ACTION)

    args = parser.parse_args()
    controller = ClassificationController(args.train)
    MainWindow(controller).run()

if __name__ == "__main__":
    main()
