from ui.interface import print_interface
from ui.user_interface import MainWindow, AuthenticationWindow, TrainingWindow, DialogWindow
from controller.controller import ClassificationController

def main():
    controller = ClassificationController()
    #print_interface(controller)
    MainWindow(controller).run()
    #AuthenticationWindow(controller).run()
    #TrainingWindow(controller).run()
    #DialogWindow('This a large text, so I hope it fits in the DialogWindow I just created').run()

if __name__ == "__main__":
    main()
