import PySimpleGUI as sg
import textwrap
import functools
import threading
from controller.controller import ClassificationController
from configs import config, uiconfig
from utils import file_management as fm

class MainWindow:

    def __init__(self, controller: ClassificationController):
        self.controller = controller
        self.uploaded_file = ''
        self.authenticated = False
        self.last_predictions = []
    
    def get_element_by_key(self, window, key):
        return window[key]

    def handle_message_text_area_event(self, window, values):
        disable_messages_file_btn = len(values[uiconfig.UI_KEY_MSG_TXT_AREA].strip()) > 0
        window[uiconfig.UI_KEY_MSG_FILE_BTN].update(disabled=disable_messages_file_btn)
        window[uiconfig.UI_KEY_CLASSIFY_BTN].update(disabled=not disable_messages_file_btn)

    def handle_method_change_event(self, window, values):
        option = values[uiconfig.UI_KEY_METHOD_COMBO]
        self.controller.change_classification_method(option)

    def handle_message_file_submission_event(self, window, values):
        file = values[uiconfig.UI_KEY_MSG_FILE_BTN]
        window[uiconfig.UI_KEY_MSG_FILE_TXT].update(value=uiconfig.UI_MESSAGE_CSV_FILE_DETECTED, visible=True)
        window[uiconfig.UI_KEY_MSG_TXT_AREA].update(disabled=True)
        window[uiconfig.UI_KEY_CLASSIFY_BTN].update(disabled=False)
        self.uploaded_file = file

    def handle_classify_event(self, window, values):
        result = []
        if(self.uploaded_file):
            result, errors = self.controller.predict_messages_in_file(self.uploaded_file)
        else:
            msg = values[uiconfig.UI_KEY_MSG_TXT_AREA]
            result, errors = self.controller.predict([msg])

        if(len(errors)):
            DialogWindow(uiconfig.UI_TITLE_ERROR, messages=errors).run()

        if(not len(self.controller.last_predictions)):
            self.handle_clear_event(window, values)
        else:
            output_msg_text_area = ''
            for pred in result:
                output_msg_text_area += pred.get_message_for_ui() + '\n'
            window[uiconfig.UI_KEY_MSG_TXT_AREA].update(value=output_msg_text_area)

            self.last_predictions = self.controller.last_predictions
            output = ''
            for pred in result:
                output += uiconfig.UI_VALID_PREDICTION_FORMAT.format(index=pred._index + 1)
                output += ': '
                output += pred.get_prediction_for_ui() + '\n'
            window[uiconfig.UI_KEY_PRED_TXT_AREA].update(value=output)

            window[uiconfig.UI_KEY_SAVE_CSV_BTN].update(disabled=False)
            window[uiconfig.UI_KEY_SAVE_TXT_BTN].update(disabled=False)
            window[uiconfig.UI_KEY_MSG_TXT_AREA].update(disabled=True)
            window[uiconfig.UI_KEY_CLASSIFY_BTN].update(disabled=True)
            window[uiconfig.UI_KEY_REFRESH_CLASSIFY_BTN].update(disabled=False)
            window[uiconfig.UI_KEY_MSG_FILE_BTN].update(disabled=True)

        if(self.authenticated and len(self.controller.last_predictions)):
            window[uiconfig.UI_KEY_CORRECT_PRED_PANEL].update(visible=True)
            values = list(map(lambda pred: pred._index + 1, self.controller.last_predictions))
            window[uiconfig.UI_KEY_N_MSG_COMBO].update(values=values)
        
    def handle_refresh_classify_event(self, window, values):
        result, _ = self.controller.redo_last_prediction()
        self.last_predictions = []
        output_msg_text_area = ''
        for pred in result:
            output_msg_text_area += pred.get_message_for_ui() + '\n'
        window[uiconfig.UI_KEY_MSG_TXT_AREA].update(value=output_msg_text_area)

        self.last_predictions = self.controller.last_predictions
        output = ''
        for pred in result:
            output += uiconfig.UI_VALID_PREDICTION_FORMAT.format(index=pred._index + 1)
            output += ': '
            output += pred.get_prediction_for_ui() + '\n'
        window[uiconfig.UI_KEY_PRED_TXT_AREA].update(value=output)

        window[uiconfig.UI_KEY_SAVE_CSV_BTN].update(disabled=False)
        window[uiconfig.UI_KEY_SAVE_TXT_BTN].update(disabled=False)
        window[uiconfig.UI_KEY_MSG_TXT_AREA].update(disabled=True)
        window[uiconfig.UI_KEY_CLASSIFY_BTN].update(disabled=True)
        window[uiconfig.UI_KEY_MSG_FILE_BTN].update(disabled=True)

        if(self.authenticated and len(self.controller.last_predictions)):
            window[uiconfig.UI_KEY_CORRECT_PRED_PANEL].update(visible=True)
            values = list(map(lambda pred: pred._index + 1, self.controller.last_predictions))
            window[uiconfig.UI_KEY_N_MSG_COMBO].update(values=values)
    
    def handle_clear_event(self, window, values):
        self.controller.clear_classification()
        self.uploaded_file = ''
        self.last_predictions = []
        window[uiconfig.UI_KEY_MSG_TXT_AREA].update(value='', disabled=False)
        window[uiconfig.UI_KEY_PRED_TXT_AREA].update(value='')
        window[uiconfig.UI_KEY_CLASSIFY_BTN].update(disabled=True)
        window[uiconfig.UI_KEY_REFRESH_CLASSIFY_BTN].update(disabled=True)
        window[uiconfig.UI_KEY_MSG_FILE_BTN].update(disabled=False)
        window[uiconfig.UI_KEY_MSG_FILE_TXT].update(visible=False)
        window[uiconfig.UI_KEY_METHOD_COMBO].update(disabled=False)
        window[uiconfig.UI_KEY_SAVE_CSV_BTN].update(disabled=True)
        window[uiconfig.UI_KEY_SAVE_TXT_BTN].update(disabled=True)
        window[uiconfig.UI_KEY_CORRECT_PRED_PANEL].update(visible=False)
        self.handle_cancel_correct_pred_event(window, values)
    
    def handle_save_csv_event(self, window, values):
        path = values[uiconfig.UI_KEY_SAVE_CSV_BTN]
        errors, filename = self.controller.save_results_to_csv(path)
        if(len(errors)):
            DialogWindow(uiconfig.UI_TITLE_ERROR_SAVE_RESULT, messages=errors).run()
        else:
            DialogWindow(uiconfig.UI_TITLE_SAVED_FILE, messages=[uiconfig.UI_MESSAGE_SAVED_FILE.format(file=filename)]).run()
        window[uiconfig.UI_KEY_SAVE_CSV_BTN].update(disabled=True)
    
    def handle_save_txt_event(self, window, values):
        path = values[uiconfig.UI_KEY_SAVE_TXT_BTN]
        errors, filename = self.controller.save_results_to_txt(path)
        if(len(errors)):
            DialogWindow(uiconfig.UI_TITLE_ERROR_SAVE_RESULT, messages=errors).run()
        else:
            DialogWindow(uiconfig.UI_TITLE_SAVED_FILE, messages=[uiconfig.UI_MESSAGE_SAVED_FILE.format(file=filename)]).run()
        window[uiconfig.UI_KEY_SAVE_TXT_BTN].update(disabled=True)

    def handle_open_auth_event(self, window, values):
        self.authenticated = AuthenticationWindow(self.controller).run()
        if(self.authenticated):
            window[uiconfig.UI_KEY_AUTH_BTN].update(visible=False)
            window[uiconfig.UI_KEY_TRAIN_BTN].update(visible=True)
            self.handle_clear_event(window, values)

    def handle_open_train_event(self, window, values):
        TrainingWindow(self.controller).run()

    def handle_n_msg_combo_event(self, window, values):
        if(self.controller.model.get_model_opt() == uiconfig.UI_BINARY_MODEL):
            window[uiconfig.UI_KEY_CORRECT_PRED_BIN_PANEL].update(visible=True)
        elif(self.controller.model.get_model_opt() == uiconfig.UI_MULTILABEL_MODEL):
            window[uiconfig.UI_KEY_FIRST_CORRECT_PRED_ML_PANEL].update(visible=True)
            window[uiconfig.UI_KEY_SECOND_CORRECT_PRED_ML_PANEL].update(visible=True)
            window[uiconfig.UI_KEY_THIRD_CORRECT_PRED_ML_PANEL].update(visible=True)
        window[uiconfig.UI_KEY_SUBMIT_CORRECT_PRED_BTN].update(disabled=False)
        

    def handle_submit_correct_pred_event(self, window, values):
        msg_index = values[uiconfig.UI_KEY_N_MSG_COMBO]
        prediction_values = []
        prediction_strings = []
        if(self.controller.model.get_model_opt() == uiconfig.UI_BINARY_MODEL):
            prediction_values.append(1 if values[uiconfig.UI_KEY_BIN_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_INAPPROPRIATE else 0)
            prediction_strings.append(values[uiconfig.UI_KEY_BIN_CORRECT_PRED_COMBO])
        elif(self.controller.model.get_model_opt() == uiconfig.UI_MULTILABEL_MODEL):
            prediction_values.append(1 if values[uiconfig.UI_KEY_TOXIC_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_TOXIC else 0)
            prediction_values.append(1 if values[uiconfig.UI_KEY_SEVERE_TOXIC_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_SEVERE_TOXIC else 0)
            prediction_values.append(1 if values[uiconfig.UI_KEY_OBSCENE_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_OBSCENE else 0)
            prediction_values.append(1 if values[uiconfig.UI_KEY_THREAT_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_THREAT else 0)
            prediction_values.append(1 if values[uiconfig.UI_KEY_INSULT_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_INSULT else 0)
            prediction_values.append(1 if values[uiconfig.UI_KEY_IDENTITY_HATE_CORRECT_PRED_COMBO] == uiconfig.UI_MESSAGE_IDENTITY_HATE else 0)
            prediction_strings.append(values[uiconfig.UI_KEY_TOXIC_CORRECT_PRED_COMBO])
            prediction_strings.append(values[uiconfig.UI_KEY_SEVERE_TOXIC_CORRECT_PRED_COMBO])
            prediction_strings.append(values[uiconfig.UI_KEY_OBSCENE_CORRECT_PRED_COMBO])
            prediction_strings.append(values[uiconfig.UI_KEY_THREAT_CORRECT_PRED_COMBO])
            prediction_strings.append(values[uiconfig.UI_KEY_INSULT_CORRECT_PRED_COMBO])
            prediction_strings.append(values[uiconfig.UI_KEY_IDENTITY_HATE_CORRECT_PRED_COMBO])

        fn = functools.partial(self.controller.correct_predictions, msg_index - 1, prediction_values)
        title = uiconfig.UI_TITLE_CORRECT_PREDICTIONS
        msg = self._message_for_confirmation_dialog(msg_index, prediction_strings)
        fn_end_msg = uiconfig.UI_MESSAGE_PRED_CORRECTED_SUCCESS
        TrainingConfirmationWindow(self.controller, fn, title, msg, fn_end_msg).run()

    def handle_cancel_correct_pred_event(self, window, values):
        window[uiconfig.UI_KEY_N_MSG_COMBO].update(value='')
        window[uiconfig.UI_KEY_BIN_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_TOXIC_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_SEVERE_TOXIC_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_OBSCENE_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_THREAT_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_INSULT_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_IDENTITY_HATE_CORRECT_PRED_COMBO].update(set_to_index=0)
        window[uiconfig.UI_KEY_CORRECT_PRED_BIN_PANEL].update(visible=False)
        window[uiconfig.UI_KEY_FIRST_CORRECT_PRED_ML_PANEL].update(visible=False)
        window[uiconfig.UI_KEY_SECOND_CORRECT_PRED_ML_PANEL].update(visible=False)
        window[uiconfig.UI_KEY_THIRD_CORRECT_PRED_ML_PANEL].update(visible=False)
        window[uiconfig.UI_KEY_SUBMIT_CORRECT_PRED_BTN].update(disabled=True)
    
    def handle_submit_help_event(self, window, values):
        path = uiconfig.HELP_SUBMIT_MESSAGES_DIR
        HelpWindow(path).run()
    
    def _message_for_confirmation_dialog(self, msg_index, new_pred):
        # result = 'Operation:\t Predict messages\n'
        pred = self._get_message_by_index(msg_index - 1)
        # result += 'Message {msg_index}:\t "{msg}"\n'.format(msg_index=msg_index, msg=pred._msg)
        # result += 'Old prediction:\t {old_value}\n'.format(old_value=pred.get_prediction_for_ui())
        # result += 'New prediction:\t {new_value}\n'.format(new_value=new_pred)
        result = uiconfig.UI_MESSAGE_CONFIRMATION_CORRECT_PREDICTION.format(index=msg_index, msg=pred._msg, 
                                                                            old_value=pred.get_prediction_for_ui(), new_value=new_pred)
        return result

    def _get_message_by_index(self, msg_index):
        result = list(filter(lambda pred: pred._index == msg_index, self.last_predictions))
        return result[0]

    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED:
                break
            if event == uiconfig.UI_KEY_MSG_TXT_AREA:
                self.handle_message_text_area_event(window, values)
            if event == uiconfig.UI_KEY_METHOD_COMBO:
                self.handle_method_change_event(window, values)
            if event == uiconfig.UI_KEY_MSG_FILE_BTN:
                self.handle_message_file_submission_event(window, values)
            if event == uiconfig.UI_KEY_CLASSIFY_BTN:
                self.handle_classify_event(window, values)
            if event == uiconfig.UI_KEY_REFRESH_CLASSIFY_BTN:
                self.handle_refresh_classify_event(window, values)
            if event == uiconfig.UI_KEY_CLEAR_BTN:
                self.handle_clear_event(window, values)
            if event == uiconfig.UI_KEY_SAVE_CSV_BTN:
                self.handle_save_csv_event(window, values)
            if event == uiconfig.UI_KEY_SAVE_TXT_BTN:
                self.handle_save_txt_event(window, values)
            if event == uiconfig.UI_KEY_AUTH_BTN:
                self.handle_open_auth_event(window, values)
            if event == uiconfig.UI_KEY_TRAIN_BTN:
                self.handle_open_train_event(window, values)
            if event == uiconfig.UI_KEY_N_MSG_COMBO:
                self.handle_n_msg_combo_event(window, values)
            if event == uiconfig.UI_KEY_SUBMIT_CORRECT_PRED_BTN:
                self.handle_submit_correct_pred_event(window, values)
            if event == uiconfig.UI_KEY_CANCEL_CORRECT_PRED_BTN:
                self.handle_cancel_correct_pred_event(window, values)
            if event == uiconfig.UI_KEY_HELP_BTN:
                self.handle_submit_help_event(window, values)


    def run(self):
        up_panel_left_panel = [
           [sg.Text(uiconfig.UI_LABEL_MSG_TXT_AREA, expand_y=False, font='25')],
           [sg.Multiline(expand_x=True, expand_y = True, enable_events=True, k=uiconfig.UI_KEY_MSG_TXT_AREA)],
           [sg.Button(uiconfig.UI_CLASSIFY_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_CLASSIFY_BTN, disabled=True, bind_return_key=True), 
            sg.Button(uiconfig.UI_CLEAR_ALL_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_CLEAR_BTN),
            sg.Button(uiconfig.UI_REFRESH_BTN, enable_events=True, k=uiconfig.UI_KEY_REFRESH_CLASSIFY_BTN, disabled=True)]
        ]

        method_submission_options_panel = [
            [sg.Combo([uiconfig.UI_BINARY_MODEL, uiconfig.UI_MULTILABEL_MODEL], default_value=uiconfig.UI_BINARY_MODEL, expand_x= True, readonly=True, enable_events=True, k=uiconfig.UI_KEY_METHOD_COMBO)],
            [sg.FileBrowse(uiconfig.UI_SUBMIT_MESSAGES_FILE_BTN, enable_events=True, k=uiconfig.UI_KEY_MSG_FILE_BTN, file_types=(("CSV", "*.csv"),))]
        ]

        help_button_panel = [
            [sg.Text(expand_y=True, visible=True)],
            [sg.Button(uiconfig.UI_HELP_BTN, font='30', k=uiconfig.UI_KEY_HELP_BTN)]
        ]

        method_submission_panel = [
            sg.Column(method_submission_options_panel),
            sg.Column(help_button_panel)
        ]

        up_panel_medium_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text(uiconfig.UI_LABEL_METHOD_COMBO, font='25')],
           method_submission_panel,
           [sg.Text(uiconfig.UI_SUBMIT_MESSAGES_TXT, k=uiconfig.UI_KEY_MSG_FILE_TXT, visible=False)],
           [sg.Text(expand_y=True, visible=False)],
        ]
        up_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Button(uiconfig.UI_AUTHENTICATE_BTN, enable_events=True, k=uiconfig.UI_KEY_AUTH_BTN)],
           [sg.Button(uiconfig.UI_TRAIN_MODEL_BTN, enable_events=True, k=uiconfig.UI_KEY_TRAIN_BTN, visible=False)],
           [sg.Text(expand_y=True, visible=False)]
        ]
        up_panel = [
            sg.Column(up_panel_left_panel, expand_x = True, expand_y = True, element_justification='c'),
            sg.Column(up_panel_medium_panel, expand_x = True, expand_y = True),
            sg.Column(up_panel_right_panel, expand_x = True, expand_y = True)
        ]

        csv_panel = [
            [sg.FolderBrowse(uiconfig.UI_SAVE_CSV_BTN, disabled=True, enable_events=True, k=uiconfig.UI_KEY_SAVE_CSV_BTN)]
        ]

        txt_panel = [
            [sg.FolderBrowse(uiconfig.UI_SAVE_TXT_BTN, disabled=True, enable_events=True, k=uiconfig.UI_KEY_SAVE_TXT_BTN)]
        ]

        save_buttons_panel = [
            sg.Column(csv_panel),
            sg.Column(txt_panel)
        ]

        down_panel_left_panel = [
           [sg.Text(uiconfig.UI_LABEL_PRED_TXT_AREA, font='25')],
           [sg.Multiline(expand_x = True, expand_y = True, disabled=True, k=uiconfig.UI_KEY_PRED_TXT_AREA)],
            save_buttons_panel
        ]

        cp_up_first = [
            [sg.Text(uiconfig.UI_LABEL_N_MESSAGE_COMBO)]
        ]

        cp_up_second = [
            [sg.Combo([0, 1], default_value=0, readonly=True, enable_events=True, key=uiconfig.UI_KEY_N_MSG_COMBO)]
        ]

        cp_up_third = [
            [sg.Button(uiconfig.UI_CORRECT_PRED_BTN, enable_events=True, k=uiconfig.UI_KEY_SUBMIT_CORRECT_PRED_BTN, disabled=True)]
        ]

        cp_up_fourth = [
            [sg.Button(uiconfig.UI_CANCEL_BTN, enable_events=True, k=uiconfig.UI_KEY_CANCEL_CORRECT_PRED_BTN)]
        ]

        cp_n_msg_panel = [
            sg.Column(cp_up_first, expand_x = True, expand_y = True),
            sg.Column(cp_up_second, expand_x = True, expand_y = True),
            sg.Column(cp_up_third, expand_x = True, expand_y = True),
            sg.Column(cp_up_fourth, expand_x = True, expand_y = True)
        ]

        cp_down_first = [
           [sg.Text(uiconfig.UI_LABEL_CORRECT_PRED_BIN_COMBO)],
           [sg.Combo([uiconfig.UI_MESSAGE_APPROPRIATE, uiconfig.UI_MESSAGE_INAPPROPRIATE], default_value=uiconfig.UI_MESSAGE_APPROPRIATE, readonly=True, key=uiconfig.UI_KEY_BIN_CORRECT_PRED_COMBO)],
        ]

        cp_down_second = [
           [sg.Text(uiconfig.UI_MESSAGE_TOXIC)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_TOXIC, uiconfig.UI_MESSAGE_TOXIC], default_value=uiconfig.UI_MESSAGE_NON_TOXIC, readonly=True, key=uiconfig.UI_KEY_TOXIC_CORRECT_PRED_COMBO)],
           [sg.Text(uiconfig.UI_MESSAGE_THREAT)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_THREAT, uiconfig.UI_MESSAGE_THREAT], default_value=uiconfig.UI_MESSAGE_NON_THREAT, readonly=True, key=uiconfig.UI_KEY_THREAT_CORRECT_PRED_COMBO)]
        ]

        cp_down_third = [
           [sg.Text(uiconfig.UI_MESSAGE_SEVERE_TOXIC)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_SEVERE_TOXIC, uiconfig.UI_MESSAGE_SEVERE_TOXIC], default_value=uiconfig.UI_MESSAGE_NON_SEVERE_TOXIC, readonly=True, key=uiconfig.UI_KEY_SEVERE_TOXIC_CORRECT_PRED_COMBO)],
           [sg.Text(uiconfig.UI_MESSAGE_INSULT)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_INSULT, uiconfig.UI_MESSAGE_INSULT], default_value=uiconfig.UI_MESSAGE_NON_INSULT, readonly=True, key=uiconfig.UI_KEY_INSULT_CORRECT_PRED_COMBO)]
        ]

        cp_down_fourth = [
           [sg.Text(uiconfig.UI_MESSAGE_OBSCENE)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_OBSCENE, uiconfig.UI_MESSAGE_OBSCENE], default_value=uiconfig.UI_MESSAGE_NON_OBSCENE, readonly=True, key=uiconfig.UI_KEY_OBSCENE_CORRECT_PRED_COMBO)],
           [sg.Text(uiconfig.UI_MESSAGE_IDENTITY_HATE)],
           [sg.Combo([uiconfig.UI_MESSAGE_NON_IDENTITY_HATE, uiconfig.UI_MESSAGE_IDENTITY_HATE], default_value=uiconfig.UI_MESSAGE_NON_IDENTITY_HATE, readonly=True, key=uiconfig.UI_KEY_IDENTITY_HATE_CORRECT_PRED_COMBO)]
        ]

        cp_down_panel = [
            sg.Column(cp_down_first, expand_x = True, expand_y = True, visible=False, k=uiconfig.UI_KEY_CORRECT_PRED_BIN_PANEL),
            sg.Column(cp_down_second, expand_x = True, expand_y = True, visible=False, k=uiconfig.UI_KEY_FIRST_CORRECT_PRED_ML_PANEL),
            sg.Column(cp_down_third, expand_x = True, expand_y = True, visible=False, k=uiconfig.UI_KEY_SECOND_CORRECT_PRED_ML_PANEL),
            sg.Column(cp_down_fourth, expand_x = True, expand_y = True, visible=False, k=uiconfig.UI_KEY_THIRD_CORRECT_PRED_ML_PANEL)
        ]

        down_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text(uiconfig.UI_LABEL_CORRECT_PRED, font='27')],
           cp_n_msg_panel,
           cp_down_panel
        ]
        down_panel = [
            sg.Column(down_panel_left_panel, expand_x = True, expand_y = True, element_justification='c'),
            sg.Column(down_panel_right_panel, expand_x = True, expand_y = True, k=uiconfig.UI_KEY_CORRECT_PRED_PANEL, visible=False)
        ]
        layout = [
            [up_panel],
            [down_panel]
        ]
        window = sg.Window(title=uiconfig.UI_TITLE_MAIN_WINDOW, layout=[layout], size=(960, 544))

        self.handle_events(window)
        window.close()

class AuthenticationWindow:
    def __init__(self, controller: ClassificationController):
        self.controller = controller
        self.authenticated = False

    def get_element_by_key(self, window, key):
        return window[key]

    def handle_submission_event(self, window, values):
        username = values[uiconfig.UI_KEY_USR_IN]
        password = values[uiconfig.UI_KEY_PWD_IN]
        errors = self.controller.authenticate(username, password)
        if(len(errors)):
            msg = self._build_errors_message(errors)
            window[uiconfig.UI_KEY_MSG_TXT].update(value=msg, visible=True)
            window[uiconfig.UI_KEY_PWD_IN].update(value='')
        else:
            self.authenticated = True

    def handle_credential_in_event(self, window, values):
        username = values[uiconfig.UI_KEY_USR_IN]
        password = values[uiconfig.UI_KEY_PWD_IN]
        disable_submit = len(username.strip()) == 0 or len(password.strip()) == 0
        window[uiconfig.UI_KEY_SUBMIT_BTN].update(disabled=disable_submit)

    def handle_events(self, window):
        while not self.authenticated:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED or event == uiconfig.UI_KEY_CANCEL_BTN:
                break
            if event == uiconfig.UI_KEY_SUBMIT_BTN:
                self.handle_submission_event(window, values)
            if event == uiconfig.UI_KEY_USR_IN:
                self.handle_credential_in_event(window, values)
            if event == uiconfig.UI_KEY_PWD_IN:
                self.handle_credential_in_event(window, values)
    
    def _build_errors_message(self, errors):
        return '\n'.join(errors)

    def run(self):
        f_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        f_center_panel = [
            [sg.Text(text_color='red', k=uiconfig.UI_KEY_MSG_TXT, visible=False)]
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
            [sg.Text(uiconfig.UI_LABEL_USR_IN)]
        ]

        s_right_panel = [
            [sg.Input(k=uiconfig.UI_KEY_USR_IN, enable_events=True, focus=True)]
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_x = True, expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        t_left_panel = [
            [sg.Text(uiconfig.UI_LABEL_PWD_IN)]
        ]

        t_right_panel = [
            [sg.Input(k=uiconfig.UI_KEY_PWD_IN, enable_events=True, password_char=uiconfig.UI_PASSWORD_CHARACTER)]
        ]

        third_panel = [
            sg.Column(t_left_panel, expand_x = True, expand_y = True),
            sg.Column(t_right_panel, expand_x = True, expand_y = True)
        ]

        fo_left_panel = [
            [sg.Button(uiconfig.UI_SUBMIT_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_SUBMIT_BTN, disabled=True, bind_return_key=True)]
        ]

        fo_right_panel = [
            [sg.Button(uiconfig.UI_CANCEL_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_CANCEL_BTN)]
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

        window = sg.Window(title=uiconfig.UI_TITLE_AUTHENTICATION_WINDOW, layout=[layout], size=(424, 240), modal=True)

        self.handle_events(window)
        window.close()
        return self.authenticated

class TrainingWindow:
    def __init__(self, controller: ClassificationController):
        self.controller = controller

    def get_element_by_key(self, window, key):
        return window[key]

    def handle_file_submission_event(self, window, values):
        file = values[uiconfig.UI_KEY_FINE_IN]
        disable_submit = len(file.strip()) == 0
        window[uiconfig.UI_KEY_SUBMIT_BTN].update(disabled=disable_submit)

    def handle_train_submission_event(self, window, values):
        model_opt = values[uiconfig.UI_KEY_METHOD_COMBO]
        file = values[uiconfig.UI_KEY_FINE_IN]
        fn = functools.partial(self.controller.train_models, model_opt, file)
        title = uiconfig.UI_TITLE_TRAINING_CONFIRMATION_WINDOW
        msg = self._message_for_confirmation_dialog(model_opt, file)
        fn_end_msg = uiconfig.UI_MESSAGE_TRAINING_SUCCESS
        TrainingConfirmationWindow(self.controller, fn, title, msg, fn_end_msg).run()
        window.write_event_value(uiconfig.UI_KEY_EXIT, None)

    
    
    def handle_method_combo_event(self, window, values):
        if(values[uiconfig.UI_KEY_METHOD_COMBO] == uiconfig.UI_BINARY_MODEL):
            window[uiconfig.UI_KEY_EXAMPLE_TXT].update(uiconfig.UI_MESSAGE_HELP_BIN_TRAIN_FILE_FORMAT)
        if(values[uiconfig.UI_KEY_METHOD_COMBO] == uiconfig.UI_MULTILABEL_MODEL):
            window[uiconfig.UI_KEY_EXAMPLE_TXT].update(uiconfig.UI_MESSAGE_HELP_ML_TRAIN_FILE_FORMAT)
    
    def handle_train_help_event(self, window, values):
        path = uiconfig.HELP_TRAIN_DATA_DIR
        HelpWindow(path).run()

    def _message_for_confirmation_dialog(self, model_opt, file):
        # result = 'Operation:\t Train model\n'
        # result += 'Model to train:\t {model_opt} model\n'.format(model_opt=model_opt)
        # result += 'Data:\t {file}'.format(file=file)
        result = uiconfig.UI_MESSAGE_CONFIRMATION_TRAIN_MODEL.format(model_opt=model_opt, file=file)
        return result

    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED or event == uiconfig.UI_KEY_CANCEL_BTN:
                break
            if event == uiconfig.UI_KEY_FINE_IN:
                self.handle_file_submission_event(window, values)
            if event == uiconfig.UI_KEY_SUBMIT_BTN:
                self.handle_train_submission_event(window, values)
            if event == uiconfig.UI_KEY_METHOD_COMBO:
                self.handle_method_combo_event(window, values)
            if event == uiconfig.UI_KEY_HELP_BTN:
                self.handle_train_help_event(window, values)

    def run(self):
        f_left_panel = [
            [sg.Text(uiconfig.UI_LABEL_METHOD_COMBO)]
        ]
        f_right_panel = [
            [sg.Combo([uiconfig.UI_BINARY_MODEL, uiconfig.UI_MULTILABEL_MODEL], default_value=uiconfig.UI_BINARY_MODEL, readonly=True, enable_events=True, key=uiconfig.UI_KEY_METHOD_COMBO)]
        ]

        first_panel = [
            sg.Column(f_left_panel, expand_x = True, expand_y = True),
            sg.Column(f_right_panel, expand_x = True, expand_y = True)
        ]

        s_left_panel = [
            [sg.Text(uiconfig.UI_LABEL_SUBMIT_TRAIN_FILE)]
        ]

        s_center_panel = [
            [sg.Input(readonly=True, enable_events=True, k=uiconfig.UI_KEY_FINE_IN)]
        ]

        s_right_panel = [
            [sg.FileBrowse(uiconfig.UI_SUBMIT_TRAIN_FILE_BTN, target=uiconfig.UI_KEY_FINE_IN, file_types=(("CSV", "*.csv"),), enable_events=True, k=uiconfig.UI_KEY_FILE_BTN)]     
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_y = True),
            sg.Column(s_center_panel, size=(250,),expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        t_left_panel = [
            [sg.Text(uiconfig.UI_MESSAGE_HELP_BIN_TRAIN_FILE_FORMAT, font='20', text_color='light gray', expand_x=True, expand_y=True, k=uiconfig.UI_KEY_EXAMPLE_TXT, justification='c')]
        ]

        third_panel = [
            sg.Column(t_left_panel, expand_y=True, expand_x=True, element_justification='c'),
            sg.Button(uiconfig.UI_HELP_BTN, font='30', k=uiconfig.UI_KEY_HELP_BTN)
        ]

        fi_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        fi_center_panel = [
            [sg.Button(uiconfig.UI_SUBMIT_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_SUBMIT_BTN, disabled=True)]
        ]

        fi_right_panel = [
            [sg.Button(uiconfig.UI_CANCEL_BTN, expand_x = True, expand_y=False, enable_events=True, k=uiconfig.UI_KEY_CANCEL_BTN)]
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
        
        window = sg.Window(title=uiconfig.UI_TITLE_TRAIN_MODEL_WINDOW, layout=[layout], size=(424, 240), modal=True)

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
        window[uiconfig.UI_KEY_CANCEL_BTN].update(visible=False)
        window[uiconfig.UI_KEY_CONFIRM_BTN].update(visible=False)
        window[uiconfig.UI_KEY_CONFIRM_TXT].update(visible=False)
        window[uiconfig.UI_KEY_OK_BTN].update(visible=True)
        window[uiconfig.UI_KEY_MSG_TXT].update(value=uiconfig.UI_MESSAGE_PERFORMING_OPERATION)
        window.refresh()
        errors = self.controller_fn()
        #window[uiconfig.UI_KEY_MSG_TXT].update(font='30')
        if(len(errors)):
            window[uiconfig.UI_KEY_MSG_TXT].update(self._build_error_message(errors))
        else:
            window[uiconfig.UI_KEY_MSG_TXT].update(self.final_msg)
        window[uiconfig.UI_KEY_OK_BTN].update(disabled=False)
    
    def handle_events(self, window):
        while True:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED or event == uiconfig.UI_KEY_CANCEL_BTN or event == uiconfig.UI_KEY_OK_BTN:
                break
            if event == uiconfig.UI_KEY_CONFIRM_BTN:
                self.handle_confirmation_event(window, values)
    
    def _build_error_message(self, errors):
        return '\n'.join(textwrap.wrap('\n'.join(errors), 125, replace_whitespace=False))

    def _get_wrapped_msg(self):
        return '\n'.join(textwrap.wrap(self.window_msg, 125, replace_whitespace=False))

    def run(self):
        elements = [
            [sg.Text(self._get_wrapped_msg(), expand_y=True, font='25', k=uiconfig.UI_KEY_MSG_TXT)],
            [sg.Text(expand_y=True, visible=False)],
            [sg.Text(uiconfig.UI_LABEL_CONFIRM_BTN, k=uiconfig.UI_KEY_CONFIRM_TXT)],
            [sg.Button(uiconfig.UI_CONFIRM_BTN, k=uiconfig.UI_KEY_CONFIRM_BTN), sg.Button(uiconfig.UI_CANCEL_BTN, k=uiconfig.UI_KEY_CANCEL_BTN), sg.Button('OK', k=uiconfig.UI_KEY_OK_BTN, visible=False, disabled=True)]
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
        return self._build_messages_message(self.messages)

    def _build_messages_message(self, messages):
        if(len(messages) > 1):
            return '\n'.join(messages)
        else:
            return '\n'.join(textwrap.wrap(''.join(messages), 60))

    def _get_messages_len(self):
        return len(self.messages)

    def run(self):
        elements = [
            [sg.Text(self._get_message(), font='25', expand_y=True, justification='c', visible= not self._get_messages_len() > 1)],
            [sg.Multiline(self._get_message(), expand_x = True, expand_y = True, visible= self._get_messages_len() > 1)],
            [sg.Button(uiconfig.UI_OK_BTN, enable_events=True, k=uiconfig.UI_KEY_OK_BTN)]
        ]

        layout = [
            sg.Column(elements, expand_x = True, expand_y = True, element_justification='c')
        ]
        
        window = sg.Window(title=self.title, layout=[layout], size=(424, 120), modal=True, finalize=True)
        while True:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED or event == uiconfig.UI_KEY_OK_BTN:
                break
        window.close()

class HelpWindow:

    def __init__(self, text_file_path):
        self.text = fm.file_to_string(text_file_path)

    def run(self):
        elements = [
            [sg.Text(self.text, expand_y=True)],
            [sg.Button(uiconfig.UI_OK_BTN, enable_events=True, k=uiconfig.UI_KEY_OK_BTN)]
        ]

        layout = [
            sg.Column(elements, expand_x = True, expand_y = True, element_justification='c')
        ]
        
        window = sg.Window(title=uiconfig.UI_TITLE_HELP_WINDOW, layout=[layout], size=(636, 330), modal=True, finalize=True)
        while True:
            event, values = window.read()
            if event == uiconfig.UI_KEY_EXIT or event == sg.WINDOW_CLOSED or event == uiconfig.UI_KEY_OK_BTN:
                break
        window.close()