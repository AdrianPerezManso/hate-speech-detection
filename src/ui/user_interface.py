import PySimpleGUI as sg

def run():
    layout = [
        sg.Text('Submit a file'),
        sg.Input(size=(25,1), enable_events=True, key='-FOLDER-'),
        sg.FileBrowse(button_text='⚙', file_types=[("CSV Files", "*.csv")], target='-FOLDER-', enable_events=True)
    ]
    window = sg.Window(title="Hello World", layout=[layout], margins=(100, 50))

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WINDOW_CLOSED:
            break
        if event == '-FOLDER-':
            file = values['-FOLDER-']
            print(type(file))
    window.close()

