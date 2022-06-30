from ui.interface import print_interface
from ui.user_interface import run
from controller.controller import Controller

def main():
    controller = Controller()
    print_interface(controller)
    #run()

if __name__ == "__main__":
    main()
