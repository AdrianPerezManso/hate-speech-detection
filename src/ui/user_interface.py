import PySimpleGUI as sg
import textwrap
import functools
import threading
from controller.controller import ClassificationController
from configs import config

class MainWindow:

    def __init__(self, controller: ClassificationController):
        self.controller = controller
        self.uploaded_file = ''
        self.authenticated = False
        self.last_predictions = []
    
    def get_element_by_key(self, window, key):
        return window[key]

    def handle_message_text_area_event(self, window, values):
        disable_messages_file_btn = len(values['<msg_text_area>'].strip()) > 0
        window['<msg_file_btn>'].update(disabled=disable_messages_file_btn)
        window['<classify_btn>'].update(disabled=not disable_messages_file_btn)

    def handle_method_change_event(self, window, values):
        option = values['<method_combo>']
        self.controller.change_classification_method(option)

    def handle_message_file_submission_event(self, window, values):
        file = values['<msg_file_btn>']
        window['<msg_file_txt>'].update(value='File .csv detected. Press Classify', visible=True)
        window['<msg_text_area>'].update(disabled=True)
        window['<classify_btn>'].update(disabled=False)
        self.uploaded_file = file

    def handle_classify_event(self, window, values):
        result = []
        if(self.uploaded_file):
            result, errors = self.controller.predict_messages_in_file(self.uploaded_file)
        else:
            msg = values['<msg_text_area>']
            result, errors = self.controller.predict([msg])

        if(len(errors)):
            DialogWindow('ERROR', messages=errors).run()

        if(not len(self.controller.last_predictions)):
            self.handle_clear_event(window, values)
        else:
            output_msg_text_area = ''
            for pred in result:
                output_msg_text_area += pred.get_message_for_ui() + '\n'
            window['<msg_text_area>'].update(value=output_msg_text_area)

            self.last_predictions = self.controller.last_predictions
            output = ''
            for pred in result:
                output += config.OUTPUT_VALID_PREDICTION_FORMAT.format(index=pred._index + 1)
                output += ': '
                output += pred.get_prediction_for_ui() + '\n'
            window['<pred_text_area>'].update(value=output)

            window['<save_csv_btn>'].update(disabled=False)
            window['<save_txt_btn>'].update(disabled=False)
            window['<method_combo>'].update(disabled=True)
            window['<msg_text_area>'].update(disabled=True)
            window['<classify_btn>'].update(disabled=True)
            window['<msg_file_btn>'].update(disabled=True)

        if(self.authenticated and len(self.controller.last_predictions)):
            window['<correct_pred_panel>'].update(visible=True)
            values = list(map(lambda pred: pred._index + 1, self.controller.last_predictions))
            window['<n_msg_combo>'].update(values=values)
    
    def handle_clear_event(self, window, values):
        self.controller.clear_classification()
        self.uploaded_file = ''
        self.last_predictions = []
        window['<msg_text_area>'].update(value='', disabled=False)
        window['<pred_text_area>'].update(value='')
        window['<classify_btn>'].update(disabled=True)
        window['<msg_file_btn>'].update(disabled=False)
        window['<msg_file_txt>'].update(visible=False)
        window['<method_combo>'].update(disabled=False)
        window['<save_csv_btn>'].update(disabled=True)
        window['<save_txt_btn>'].update(disabled=True)
        window['<correct_pred_panel>'].update(visible=False)
        self.handle_cancel_correct_pred_event(window, values)
    
    def handle_save_csv_event(self, window, values):
        path = values['<save_csv_btn>']
        errors, filename = self.controller.save_results_to_csv(path)
        if(len(errors)):
            DialogWindow('ERROR: Save result', messages=errors).run()
        else:
            DialogWindow('Saved file', messages=['Saved {file}'.format(file=filename)]).run()
        window['<save_csv_btn>'].update(disabled=True)
    
    def handle_save_txt_event(self, window, values):
        path = values['<save_txt_btn>']
        errors, filename = self.controller.save_results_to_txt(path)
        if(len(errors)):
            DialogWindow('ERROR: Save result', messages=errors).run()
        else:
            DialogWindow('Saved file', messages=['Saved {file}'.format(file=filename)]).run()
        window['<save_txt_btn>'].update(disabled=True)

    def handle_open_auth_event(self, window, values):
        self.authenticated = AuthenticationWindow(self.controller).run()
        if(self.authenticated):
            window['<auth_btn>'].update(visible=False)
            window['<train_btn>'].update(visible=True)
            self.handle_clear_event(window, values)

    def handle_open_train_event(self, window, values):
        TrainingWindow(self.controller).run()

    def handle_n_msg_combo_event(self, window, values):
        if(self.controller.model.get_model_opt() == 'Binary'):
            window['<correct_pred_panel_binary>'].update(visible=True)
        elif(self.controller.model.get_model_opt() == 'Itemized'):
            window['<first_correct_pred_panel_ml>'].update(visible=True)
            window['<second_correct_pred_panel_ml>'].update(visible=True)
            window['<third_correct_pred_panel_ml>'].update(visible=True)
        window['<submit_correct_pred_btn>'].update(disabled=False)
        

    def handle_submit_correct_pred_event(self, window, values):
        msg_index = values['<n_msg_combo>']
        prediction_values = []
        prediction_strings = []
        if(self.controller.model.get_model_opt() == 'Binary'):
            prediction_values.append(1 if values['<bin_correct_pred_combo>'] == 'Inappropriate' else 0)
            prediction_strings.append(values['<bin_correct_pred_combo>'])
        elif(self.controller.model.get_model_opt() == 'Itemized'):
            prediction_values.append(1 if values['<toxic_correct_pred_combo>'] == 'Toxic' else 0)
            prediction_values.append(1 if values['<severe_toxic_correct_pred_combo>'] == 'Severe toxic' else 0)
            prediction_values.append(1 if values['<obscene_correct_pred_combo>'] == 'Obscene' else 0)
            prediction_values.append(1 if values['<threat_correct_pred_combo>'] == 'Threat' else 0)
            prediction_values.append(1 if values['<insult_correct_pred_combo>'] == 'Insult' else 0)
            prediction_values.append(1 if values['<identity_hate_correct_pred_combo>'] == 'Identity hate' else 0)
            prediction_strings.append(values['<toxic_correct_pred_combo>'])
            prediction_strings.append(values['<severe_toxic_correct_pred_combo>'])
            prediction_strings.append(values['<obscene_correct_pred_combo>'])
            prediction_strings.append(values['<threat_correct_pred_combo>'])
            prediction_strings.append(values['<insult_correct_pred_combo>'])
            prediction_strings.append(values['<identity_hate_correct_pred_combo>'])

        fn = functools.partial(self.controller.correct_predictions, msg_index - 1, prediction_values)
        title = 'Correcting predictions'
        msg = self._message_for_confirmation_dialog(msg_index, prediction_strings)
        fn_end_msg = 'Prediction successfully corrected'
        TrainingConfirmationWindow(self.controller, fn, title, msg, fn_end_msg).run()


    def handle_cancel_correct_pred_event(self, window, values):
        window['<n_msg_combo>'].update(value='')
        window['<bin_correct_pred_combo>'].update(set_to_index=0)
        window['<toxic_correct_pred_combo>'].update(set_to_index=0)
        window['<severe_toxic_correct_pred_combo>'].update(set_to_index=0)
        window['<obscene_correct_pred_combo>'].update(set_to_index=0)
        window['<threat_correct_pred_combo>'].update(set_to_index=0)
        window['<insult_correct_pred_combo>'].update(set_to_index=0)
        window['<identity_hate_correct_pred_combo>'].update(set_to_index=0)
        window['<correct_pred_panel_binary>'].update(visible=False)
        window['<first_correct_pred_panel_ml>'].update(visible=False)
        window['<second_correct_pred_panel_ml>'].update(visible=False)
        window['<third_correct_pred_panel_ml>'].update(visible=False)
        window['<submit_correct_pred_btn>'].update(disabled=True)
    
    def _message_for_confirmation_dialog(self, msg_index, new_pred):
        result = 'Operation:\t Predict messages\n'
        pred = self._get_message_by_index(msg_index - 1)
        result += 'Message {msg_index}:\t "{msg}"\n'.format(msg_index=msg_index, msg=pred._msg)
        result += 'Old prediction:\t {old_value}\n'.format(old_value=pred.get_prediction_for_ui())
        result += 'New prediction:\t {new_value}\n'.format(new_value=new_pred)
        return result

    def _get_message_by_index(self, msg_index):
        result = list(filter(lambda pred: pred._index == msg_index, self.last_predictions))
        return result[0]

    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED:
                break
            if event == '<msg_text_area>':
                self.handle_message_text_area_event(window, values)
            if event == '<method_combo>':
                self.handle_method_change_event(window, values)
            if event == '<msg_file_btn>':
                self.handle_message_file_submission_event(window, values)
            if event == '<classify_btn>':
                self.handle_classify_event(window, values)
            if event == '<clear_btn>':
                self.handle_clear_event(window, values)
            if event == '<save_csv_btn>':
                self.handle_save_csv_event(window, values)
            if event == '<save_txt_btn>':
                self.handle_save_txt_event(window, values)
            if event == '<auth_btn>':
                self.handle_open_auth_event(window, values)
            if event == '<train_btn>':
                self.handle_open_train_event(window, values)
            if event == '<n_msg_combo>':
                self.handle_n_msg_combo_event(window, values)
            if event == '<submit_correct_pred_btn>':
                self.handle_submit_correct_pred_event(window, values)
            if event == '<cancel_correct_pred_btn>':
                self.handle_cancel_correct_pred_event(window, values)


    def run(self):
        up_panel_left_panel = [
           [sg.Text('Write a message', expand_y=False, font='25')],
           [sg.Multiline(expand_x=True, expand_y = True, enable_events=True, k='<msg_text_area>')],
           [sg.Button('Classify', expand_x = True, expand_y=False, enable_events=True, k='<classify_btn>', disabled=True, bind_return_key=True), 
            sg.Button('Clear All', expand_x = True, expand_y=False, enable_events=True, k='<clear_btn>')]
        ]
        up_panel_medium_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text('Select classification method', font='25')],
           [sg.Combo(['Binary', 'Itemized'], default_value='Binary', readonly=True, enable_events=True, k='<method_combo>')],
           [sg.FileBrowse('Submit messages file', enable_events=True, k='<msg_file_btn>', file_types=(("CSV", "*.csv"),))],
           [sg.Text('...Submitted data.csv', k='<msg_file_txt>', visible=False)],
           [sg.Text(expand_y=True, visible=False)],
        ]
        up_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Button('Log as administrator', enable_events=True, k='<auth_btn>')],
           [sg.Button('Train model', enable_events=True, k='<train_btn>', visible=False)],
           [sg.Text(expand_y=True, visible=False)]
        ]
        up_panel = [
            sg.Column(up_panel_left_panel, expand_x = True, expand_y = True),
            sg.Column(up_panel_medium_panel, expand_x = True, expand_y = True),
            sg.Column(up_panel_right_panel, expand_x = True, expand_y = True)
        ]
        down_panel_left_panel = [
           [sg.Text('Prediction', font='25')],
           [sg.Multiline(expand_x = True, expand_y = True, disabled=True, k='<pred_text_area>')]
        ]
        down_panel_medium_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.FolderBrowse('Save results to .csv file', disabled=True, enable_events=True, k='<save_csv_btn>')],
           [sg.FolderBrowse('Save results to .txt file', disabled=True, enable_events=True, k='<save_txt_btn>')],
           [sg.Text(expand_y=True, visible=False)],
        ]

        cp_up_first = [
            [sg.Text('Nº of the message:')]
        ]

        cp_up_second = [
            [sg.Combo([0, 1], default_value=0, readonly=True, enable_events=True, key='<n_msg_combo>')]
        ]

        cp_up_third = [
            [sg.Button('Correct', enable_events=True, k='<submit_correct_pred_btn>', disabled=True)]
        ]

        cp_up_fourth = [
            [sg.Button('Cancel', enable_events=True, k='<cancel_correct_pred_btn>')]
        ]

        cp_n_msg_panel = [
            sg.Column(cp_up_first, expand_x = True, expand_y = True),
            sg.Column(cp_up_second, expand_x = True, expand_y = True),
            sg.Column(cp_up_third, expand_x = True, expand_y = True),
            sg.Column(cp_up_fourth, expand_x = True, expand_y = True)
        ]

        cp_down_first = [
           [sg.Text('Inappropriate:')],
           [sg.Combo(['Appropriate', 'Inappropriate'], default_value='Appropriate', readonly=True, key='<bin_correct_pred_combo>')],
        ]

        cp_down_second = [
           [sg.Text('Toxic:')],
           [sg.Combo(['Not toxic', 'Toxic'], default_value='Not toxic', readonly=True, key='<toxic_correct_pred_combo>')],
           [sg.Text('Threat:')],
           [sg.Combo(['Not threat', 'Threat'], default_value='Not threat', readonly=True, key='<threat_correct_pred_combo>')]
        ]

        cp_down_third = [
           [sg.Text('Severe toxic:')],
           [sg.Combo(['Not severe toxic', 'Severe toxic'], default_value='Not severe toxic', readonly=True, key='<severe_toxic_correct_pred_combo>')],
           [sg.Text('Insult:')],
           [sg.Combo(['Not insult', 'Insult'], default_value='Not insult', readonly=True, key='<insult_correct_pred_combo>')]
        ]

        cp_down_fourth = [
           [sg.Text('Obscene:')],
           [sg.Combo(['Not obscene', 'Obscene'], default_value='Not obscene', readonly=True, key='<obscene_correct_pred_combo>')],
           [sg.Text('Identity hate:')],
           [sg.Combo(['Not identity hate', 'Identity hate'], default_value='Not identity hate', readonly=True, key='<identity_hate_correct_pred_combo>')]
        ]

        cp_down_panel = [
            sg.Column(cp_down_first, expand_x = True, expand_y = True, visible=False, k='<correct_pred_panel_binary>'),
            sg.Column(cp_down_second, expand_x = True, expand_y = True, visible=False, k='<first_correct_pred_panel_ml>'),
            sg.Column(cp_down_third, expand_x = True, expand_y = True, visible=False, k='<second_correct_pred_panel_ml>'),
            sg.Column(cp_down_fourth, expand_x = True, expand_y = True, visible=False, k='<third_correct_pred_panel_ml>')
        ]

        down_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text('Correct predictions', font='27')],
           cp_n_msg_panel,
           cp_down_panel
        ]
        down_panel = [
            sg.Column(down_panel_left_panel, expand_x = True, expand_y = True),
            sg.Column(down_panel_medium_panel, expand_x = True, expand_y = True),
            sg.Column(down_panel_right_panel, expand_x = True, expand_y = True, k='<correct_pred_panel>', visible=False)
        ]
        layout = [
            [up_panel],
            [down_panel]
        ]
        window = sg.Window(title="Detect inappropriate messages", layout=[layout], size=(960, 544))

        self.handle_events(window)
        window.close()

class AuthenticationWindow:
    def __init__(self, controller: ClassificationController):
        self.controller = controller
        self.authenticated = False

    def get_element_by_key(self, window, key):
        return window[key]

    def handle_submission_event(self, window, values):
        username = values['<usr_in>']
        password = values['<pwd_in>']
        errors = self.controller.authenticate(username, password)
        if(len(errors)):
            msg = self._build_errors_message(errors)
            window['<msg_txt>'].update(value=msg, visible=True)
            window['<pwd_in>'].update(value='')
        else:
            self.authenticated = True

    def handle_credential_in_event(self, window, values):
        username = values['<usr_in>']
        password = values['<pwd_in>']
        disable_submit = len(username.strip()) == 0 or len(password.strip()) == 0
        window['<submit_btn>'].update(disabled=disable_submit)

    def handle_events(self, window):
        while not self.authenticated:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED or event == '<cancel_btn>':
                break
            if event == '<submit_btn>':
                self.handle_submission_event(window, values)
            if event == '<usr_in>':
                self.handle_credential_in_event(window, values)
            if event == '<pwd_in>':
                self.handle_credential_in_event(window, values)
    
    def _build_errors_message(self, errors):
        return '\n'.join(errors)

    def run(self):
        f_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        f_center_panel = [
            [sg.Text('Message for the user', text_color='red', k='<msg_txt>', visible=False)]
        ]

        f_right_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        first_panel = [
            sg.Column(f_left_panel, expand_x = True, expand_y = True),
            sg.Column(f_center_panel, expand_x = True, expand_y = True),
            sg.Column(f_right_panel, expand_x = True, expand_y = True)
        ]


        s_left_panel = [
            [sg.Text('Username:')]
        ]

        s_right_panel = [
            [sg.Input(k='<usr_in>', enable_events=True, focus=True)]
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_x = True, expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        t_left_panel = [
            [sg.Text('Password:')]
        ]

        t_right_panel = [
            [sg.Input(k='<pwd_in>', enable_events=True, password_char='*')]
        ]

        third_panel = [
            sg.Column(t_left_panel, expand_x = True, expand_y = True),
            sg.Column(t_right_panel, expand_x = True, expand_y = True)
        ]

        fo_left_panel = [
            [sg.Button('Submit', expand_x = True, expand_y=False, enable_events=True, k='<submit_btn>', disabled=True, bind_return_key=True)]
        ]

        fo_right_panel = [
            [sg.Button('Cancel', expand_x = True, expand_y=False, enable_events=True, k='<cancel_btn>')]
        ]

        fourth_panel = [
            sg.Column(fo_left_panel, expand_x = True, expand_y = True),
            sg.Column(fo_right_panel, expand_x = True, expand_y = True)
        ]

        layout = [
            [first_panel],
            [second_panel],
            [third_panel],
            [fourth_panel]
        ]

        window = sg.Window(title="Log as Administrator", layout=[layout], size=(424, 240), modal=True)

        self.handle_events(window)
        window.close()
        return self.authenticated

class TrainingWindow:
    def __init__(self, controller: ClassificationController):
        self.controller = controller

    def get_element_by_key(self, window, key):
        return window[key]

    def handle_file_submission_event(self, window, values):
        file = values['<file_in>']
        disable_submit = len(file.strip()) == 0
        window['<submit_btn>'].update(disabled=disable_submit)

    def handle_train_submission_event(self, window, values):
        model_opt = values['<method_combo>']
        file = values['<file_in>']
        fn = functools.partial(self.controller.train_models, model_opt, file)
        title = 'Training model'
        msg = self._message_for_confirmation_dialog(model_opt, file)
        fn_end_msg = 'Training was performed successfully'
        TrainingConfirmationWindow(self.controller, fn, title, msg, fn_end_msg).run()
        window.write_event_value('Exit', None)

    def _message_for_confirmation_dialog(self, model_opt, file):
        result = 'Operation:\t Train model\n'
        result += 'Model to train:\t {model_opt} model\n'.format(model_opt=model_opt)
        result += 'Data:\t {file}'.format(file=file)
        return result
    
    def handle_method_combo_event(self, window, values):
        if(values['<method_combo>'] == 'Binary'):
            window['<example_txt>'].update('Example for a row in the file: 0,”he was a boy”')
        if(values['<method_combo>'] == 'Itemized'):
            window['<example_txt>'].update('Example for a row in the file: “this is a message”,0,0,0,0,0,0')

    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED or event == '<cancel_btn>':
                break
            if event == '<file_in>':
                self.handle_file_submission_event(window, values)
            if event == '<submit_btn>':
                self.handle_train_submission_event(window, values)
            if event == '<method_combo>':
                self.handle_method_combo_event(window, values)

    def run(self):
        f_left_panel = [
            [sg.Text('Select classification method:')]
        ]
        f_right_panel = [
            [sg.Combo(['Binary', 'Itemized'], default_value='Binary', readonly=True, enable_events=True, key='<method_combo>')]
        ]

        first_panel = [
            sg.Column(f_left_panel, expand_x = True, expand_y = True),
            sg.Column(f_right_panel, expand_x = True, expand_y = True)
        ]

        s_left_panel = [
            [sg.Text('Upload file:')]
        ]

        s_center_panel = [
            [sg.Input(readonly=True, enable_events=True, k='<file_in>')]
        ]

        s_right_panel = [
            [sg.FileBrowse('⚙', target='<file_in>', file_types=(("CSV", "*.csv"),), enable_events=True, k='<file_btn>')]     
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_y = True),
            sg.Column(s_center_panel, size=(250,),expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        third_panel = [
            sg.Text('Example for a row in the file: 0,”he was a boy”', text_color='light gray', expand_y=True, expand_x=True, justification='c', k='<example_txt>')
        ]

        fi_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        fi_center_panel = [
            [sg.Button('Submit', expand_x = True, expand_y=False, enable_events=True, k='<submit_btn>', disabled=True)]
        ]

        fi_right_panel = [
            [sg.Button('Cancel', expand_x = True, expand_y=False, enable_events=True, k='<cancel_btn>')]
        ]

        fourth_panel = [
            sg.Column(fi_left_panel, expand_x = True, expand_y = True),
            sg.Column(fi_center_panel, expand_x = True, expand_y = True),
            sg.Column(fi_right_panel, expand_x = True, expand_y = True)
        ]

        layout = [
            [first_panel],
            [second_panel],
            [third_panel],
            [fourth_panel]
        ]
        
        window = sg.Window(title="Train", layout=[layout], size=(424, 240), modal=True)

        self.handle_events(window)
            
        window.close()

class TrainingConfirmationWindow:
    def __init__(self, controller, controller_fn, title, window_msg, final_msg):
        self.controller = controller
        self.controller_fn = controller_fn
        self.window_msg = window_msg
        self.title = title
        self.final_msg = final_msg

    def get_element_by_key(self, window, key):
        return window[key]

    def handle_confirmation_event(self, window, values):
        window['<cancel_btn>'].update(visible=False)
        window['<confirm_btn>'].update(visible=False)
        window['<confirm_txt>'].update(visible=False)
        window['<ok_btn>'].update(visible=True)
        window['<msg_txt>'].update(value='Performing operation. This might take some time, please wait')
        window.refresh()
        errors = self.controller_fn()
        #window['<msg_txt>'].update(font='30')
        if(len(errors)):
            window['<msg_txt>'].update(self._build_error_message(errors))
        else:
            window['<msg_txt>'].update(self.final_msg)
        window['<ok_btn>'].update(disabled=False)
    
    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED or event == '<cancel_btn>' or event == '<ok_btn>':
                break
            if event == '<confirm_btn>':
                self.handle_confirmation_event(window, values)
    
    def _build_error_message(self, errors):
        return '\n'.join(textwrap.wrap('\n'.join(errors), 125, replace_whitespace=False))

    def _get_wrapped_msg(self):
        return '\n'.join(textwrap.wrap(self.window_msg, 125, replace_whitespace=False))

    def run(self):
        elements = [
            [sg.Text(self._get_wrapped_msg(), expand_y=True, font='25', k='<msg_txt>')],
            [sg.Text(expand_y=True, visible=False)],
            [sg.Text('Proceed with operation?', k='<confirm_txt>')],
            [sg.Button('Confirm', k='<confirm_btn>'), sg.Button('Cancel', k='<cancel_btn>'), sg.Button('OK', k='<ok_btn>', visible=False, disabled=True)]
        ]

        layout = [
            sg.Column(elements, expand_x = True, expand_y = True, element_justification='c')
        ]
        window = sg.Window(title=self.title, layout=[layout], size=(636, 240), modal=True, finalize=True)
        self.handle_events(window)
        window.close()

class DialogWindow:
    def __init__(self, title, messages=[]):
        self.title = title
        self.messages = messages       

    def get_element_by_key(self, window, key):
        return window[key]

    def _get_message(self):
        return self._build_messages_message(self.messages) if len(self.messages) else 'Processing...'

    def _build_messages_message(self, messages):
        if(len(messages) > 1):
            return '\n'.join(messages)
        else:
            return '\n'.join(textwrap.wrap(''.join(messages), 60))

    def _get_messages_len(self):
        return len(self.messages)

    def run(self):
        elements = [
            [sg.Text(self._get_message(), font='25', expand_y=True, justification='c', k='<txt>', visible= not self._get_messages_len() > 1)],
            [sg.Multiline(self._get_message(), expand_x = True, expand_y = True, k='<error_txt_area>', visible= self._get_messages_len() > 1)],
            [sg.Button('OK', enable_events=True, k='<ok_btn>')]
        ]

        layout = [
            sg.Column(elements, expand_x = True, expand_y = True, element_justification='c')
        ]
        
        window = sg.Window(title=self.title, layout=[layout], size=(424, 120), modal=True, finalize=True)
        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED or event == '<ok_btn>':
                break
        window.close()