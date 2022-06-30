from controller.controller import Controller
from classifiers.classifiers import BinaryModel, MLModel

def print_interface(controller: Controller):
    while(True):
        print('1 - Predecir mensaje')
        print('2 - Entrenar modelo')
        print('3 - Cambiar modelo')
        print('4 - Autenticarse')
        print('5 - Subir archivo')
        print('9 - Salir')
        user_input = input('Opción: ')
        match user_input:
            case '1':
                msg = input('Mensaje a predecir: ')
                result = controller.predict(msg)
                print('El mensaje ha sido predecido como: ', result) if result else print('No prediction')
            case '2':
                print('Entrenando modelo')
            case '3':        
                print('1 - Modelo binario')
                print('2 - Modelo multietiqueta')
                input_modelo = input('Opción: ')
                modelo = BinaryModel() if input_modelo == '1' else MLModel()
                controller.change_classification_method(modelo)
            case '4':
                input_usr = input('Username: ')
                input_pwd = input('Password: ')
                controller.authenticate(input_usr, input_pwd)
            case '5':
                input_file_path = input('Path to file: ')
                result = controller.predict_messages(input_file_path)
                print('Los mensajes han sido predecidos como: ', result) if result else print('No prediction')
            case '9':
                return

