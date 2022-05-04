from turtle import goto


from app import App
from models import BinaryModel, MLModel

def print_interface(app: App):
    while(True):
        print('1 - Predecir mensaje')
        print('2 - Entrenar modelo')
        print('3 - Cambiar modelo')
        print('4 - Salir')
        user_input = input('Opción: ')
        match user_input:
            case '1':
                msg = input('Mensaje a predecir: ')
                print('El mensaje ha sido predecido como: ' +  app.predict(msg))
            case '2':
                print('Entrenando modelo')
            case '3':        
                print('1 - Modelo binario')
                print('2 - Modelo multietiqueta')
                input_modelo = input('Opción: ')
                modelo = BinaryModel() if input_modelo == '1' else MLModel()
                app.setModel(modelo)
            case '4':
                return

