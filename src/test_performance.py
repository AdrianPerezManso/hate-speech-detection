from controller.controller import ClassificationController
import time

from src.configs import config

def test_1_1_1_2():
    ClassificationController(True)

def test_1_3():
    messages = ['a message for performance testing' for i in range(10000)]
    controller = ClassificationController()
    start = time.time()
    controller.predict(messages)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_4():
    messages = ['a message for performance testing' for i in range(10000)]
    controller = ClassificationController()
    controller.change_classification_method('Itemized')
    start = time.time()
    controller.predict(messages)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_5():
    path = config.PROJECT_ROOT + '/testfiles/performance/' + 'messages.csv'
    controller = ClassificationController()
    start = time.time()
    controller.predict_messages_in_file(path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_6():
    path = config.PROJECT_ROOT + '/testfiles/performance/' + 'messages.csv'
    controller = ClassificationController()
    controller.change_classification_method('Itemized')
    start = time.time()
    controller.predict_messages_in_file(path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_7():
    controller = ClassificationController()
    messages = ['a message for performance testing' for i in range(10000)]
    controller.predict(messages)
    start = time.time()
    path = config.PROJECT_ROOT + '/out/'
    controller.save_results_to_csv(path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_8():
    controller = ClassificationController()
    messages = ['a message for performance testing' for i in range(10000)]
    controller.predict(messages)
    start = time.time()
    path = config.PROJECT_ROOT + '/out/'
    controller.save_results_to_txt(path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_9():
    controller = ClassificationController()
    controller.authenticate('admin1', 'admin1')
    path = config.PROJECT_ROOT + '/testfiles/performance/' + 'new_data.csv'
    start = time.time()
    controller.train_models('Binary', path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

def test_1_10():
    controller = ClassificationController()
    controller.authenticate('admin1', 'admin1')
    path = config.PROJECT_ROOT + '/testfiles/performance/' + 'new_data_ml.csv'
    start = time.time()
    controller.train_models('Itemized', path)
    end = time.time()
    e_time = end - start
    print('Execution time: ', e_time)

if __name__ == '__main__':
    #test_1_1_1_2()
    #test_1_3()
    #test_1_4()
    #test_1_5()
    #test_1_6()
    #test_1_7()
    #test_1_8()
    test_1_9()
    # test_1_10()