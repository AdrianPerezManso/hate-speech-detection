from ui.interface import print_interface
from ui.user_interface import MainWindow, AuthenticationWindow, TrainingWindow
from controller.controller import ClassificationController

def main():
    controller = ClassificationController()
    print_interface(controller)
    #MainWindow(controller).run()
    #AuthenticationWindow(controller).run()
    #TrainingWindow(controller).run()

if __name__ == "__main__":
    main()
