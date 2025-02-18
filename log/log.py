import json


class Log:

    def __init__(self):
        self.data_dict = dict()

    def write(self, key, value):

        if not (key in self.data_dict):
            self.data_dict[key] = list()

        self.data_dict[key].append(value)

    def save(self, fn):
        with open(fn, 'w') as json_file:
            json.dump(self.data_dict, json_file)

    def load(self, fn):
        with open(fn, 'r') as json_file:
            self.data_dict = json.load(json_file)
