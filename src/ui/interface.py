from controller.controller import ClassificationController
from classifiers.classifiers import BinaryModel, MLModel

def print_interface(controller: ClassificationController):
    while(True):
        print('1 - Predecir mensaje')
        print('2 - Entrenar modelo')
        print('3 - Cambiar modelo')
        print('4 - Autenticarse')
        print('5 - Subir archivo')
        print('6 - Correct prediction')
        print('7 - Save results to file')
        print('9 - Salir')
        user_input = input('Opción: ')
        match user_input:
            case '1':
                msg = input('Mensaje a predecir: ')
                result = controller.predict(msg)
                print('El mensaje ha sido predecido como: ', result) if result else print('No prediction')
            case '2':
                print('1 - Modelo binario')
                print('2 - Modelo multietiqueta')
                input_modelo = input('Opción modelo: ')
                input_file_path = input('Path to file: ')
                controller.train_models(input_modelo, input_file_path)
            case '3':        
                print('1 - Modelo binario')
                print('2 - Modelo multietiqueta')
                input_modelo = input('Opción: ')
                model = 'Binary' if input_modelo == 1 else 'Itemized'
                controller.change_classification_method(model)
            case '4':
                input_usr = input('Username: ')
                input_pwd = input('Password: ')
                controller.authenticate(input_usr, input_pwd)
            case '5':
                input_file_path = input('Path to file: ')
                result = controller.predict_messages_in_file(input_file_path)
                print('Los mensajes han sido predecidos como: ', result) if result else print('No prediction')
            case '6':
                result = []
                input_index_value = int(input('Number of the message to correct: '))
                input_prediction_value = input('New value: ')
                for char in input_prediction_value.split(','):
                    result.append(int(char))
                controller.correct_predictions(input_index_value, result)
            case '7':
                controller.save_results_to_file()
            case '9':
                return

