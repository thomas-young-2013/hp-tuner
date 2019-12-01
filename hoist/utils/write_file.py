import pickle


def write_to_file(file, data):
    file = open(file, 'wb')
    pickle.dump(data, file)
    file.close()


def load_data_from_file(file):
    file = open(file, 'rb')
    return pickle.load(file)
