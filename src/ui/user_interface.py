import PySimpleGUI as sg

from controller.controller import ClassificationController

class MainWindow:

    def __init__(self, controller: ClassificationController):
        self.controller = controller

    def run(self):
        up_panel_left_panel = [
           [sg.Text('Write a message', expand_y=False)],
           [sg.Multiline(expand_x=True, expand_y = True)],
           [sg.Button('Classify', expand_x = True, expand_y=False), sg.Button('Clear All', expand_x = True, expand_y=False)]
        ]
        up_panel_medium_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text('Select classification method')],
           [sg.Combo(['Binary', 'Itemized'], default_value='Binary', readonly=True, key='-COMBO-')],
           [sg.Button('Submit messages file')],
           [sg.Text('...Submitted data.csv')],
           [sg.Text(expand_y=True, visible=False)],
        ]
        up_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Button('Authenticate')],
           [sg.Button('Train model', disabled=True)],
           [sg.Text(expand_y=True, visible=False)]
        ]
        up_panel = [
            sg.Column(up_panel_left_panel, expand_x = True, expand_y = True),
            sg.Column(up_panel_medium_panel, expand_x = True, expand_y = True),
            sg.Column(up_panel_right_panel, expand_x = True, expand_y = True)
        ]
        down_panel_left_panel = [
           [sg.Text('Prediction')],
           [sg.Multiline(expand_x = True, expand_y = True)]
        ]
        down_panel_medium_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Button('Save results to file', disabled=True)],
           [sg.Text(expand_y=True, visible=False)],
        ]

        cp_left = [
           [sg.Text('Toxic:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)],
           [sg.Text('Threat:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)]
        ]

        cp_center = [
           [sg.Text('Severe toxic:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)],
           [sg.Text('Insult:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)]
        ]

        cp_right = [
           [sg.Text('Obscene:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)],
           [sg.Text('Identity hate:')],
           [sg.Combo([0, 1], default_value=0, readonly=True, key='-COMBO1-', disabled=True)]
        ]

        cp_panel = [
            sg.Column(cp_left, expand_x = True, expand_y = True),
            sg.Column(cp_center, expand_x = True, expand_y = True),
            sg.Column(cp_right, expand_x = True, expand_y = True)
        ]

        down_panel_right_panel = [
           [sg.Text(expand_y=True, visible=False)],
           [sg.Text('Correct predictions:')],
           cp_panel
        ]
        down_panel = [
            sg.Column(down_panel_left_panel, expand_x = True, expand_y = True),
            sg.Column(down_panel_medium_panel, expand_x = True, expand_y = True),
            sg.Column(down_panel_right_panel, expand_x = True, expand_y = True)
        ]
        layout = [
            [up_panel],
            [down_panel]
        ]
        window = sg.Window(title="Detect inappropriate message", layout=[layout], size=(848, 480))

        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED:
                break
            if event == '-BTN-':
                val = values['-COMBO-']
                print(type(val))
        window.close()

class AuthenticationWindow:
    def __init__(self, controller: ClassificationController):
        self.controller = controller

    def run(self):
        
        f_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        f_center_panel = [
            [sg.Text('Message for the user', text_color='red')]
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
            [sg.Input()]
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_x = True, expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        t_left_panel = [
            [sg.Text('Password:')]
        ]

        t_right_panel = [
            [sg.Input()]
        ]

        third_panel = [
            sg.Column(t_left_panel, expand_x = True, expand_y = True),
            sg.Column(t_right_panel, expand_x = True, expand_y = True)
        ]

        fo_left_panel = [
            [sg.Button('Submit', expand_x = True, expand_y=False)]
        ]

        fo_right_panel = [
            [sg.Button('Cancel', expand_x = True, expand_y=False)]
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

        window = sg.Window(title="Authentication", layout=[layout], size=(424, 240))

        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED:
                break
            if event == '-BTN-':
                val = values['-COMBO-']
                print(type(val))
        window.close()

class TrainingWindow:
    def __init__(self, controller: ClassificationController):
            self.controller = controller

    def run(self):
        f_left_panel = [
            [sg.Text('Select classification method:')]
        ]
        f_right_panel = [
            [sg.Combo(['Binary', 'Itemized'], default_value='Binary', readonly=True, key='-COMBO-')]
        ]

        first_panel = [
            sg.Column(f_left_panel, expand_x = True, expand_y = True),
            sg.Column(f_right_panel, expand_x = True, expand_y = True)
        ]

        s_left_panel = [
            [sg.Text('Upload file:')]
        ]

        s_center_panel = [
            [sg.Input(k='inp')]
        ]

        s_right_panel = [
            [sg.FileBrowse('⚙', target='inp', file_types=(("CSV", "*.csv"),))]     
        ]

        second_panel = [
            sg.Column(s_left_panel, expand_y = True),
            sg.Column(s_center_panel, size=(250,),expand_y = True),
            sg.Column(s_right_panel, expand_x = True, expand_y = True)
        ]

        t_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        t_center_panel = [
            [sg.Text('Message for the user')]
        ]

        t_right_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]    

        third_panel = [
            sg.Column(t_left_panel, expand_x = True, expand_y = True),
            sg.Column(t_center_panel, expand_x = True, expand_y = True),
            sg.Column(t_right_panel, expand_x = True, expand_y = True)
        ]

        fo_left_panel = [
            [sg.Text(expand_y=True, visible=False)]
        ]

        fo_center_panel = [
            [sg.Button('Submit', expand_x = True, expand_y=False)]
        ]

        fo_right_panel = [
            [sg.Button('Cancel', expand_x = True, expand_y=False)]
        ]

        fourth_panel = [
            sg.Column(fo_left_panel, expand_x = True, expand_y = True),
            sg.Column(fo_center_panel, expand_x = True, expand_y = True),
            sg.Column(fo_right_panel, expand_x = True, expand_y = True)
        ]

        layout = [
            [first_panel],
            [second_panel],
            [third_panel],
            [fourth_panel]
        ]
        
        window = sg.Window(title="Authentication", layout=[layout], size=(424, 240))

        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WINDOW_CLOSED:
                break
            if event == '-BTN-':
                val = values['-COMBO-']
                print(type(val))
        window.close()