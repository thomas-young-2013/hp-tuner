import re


def is_float(element):
    if re.match("^\d+?\.\d+?$", element) is None: return False
    else: return True


class Properties:
    def __init__(self, fileName):
        self.fileName = fileName
        self.properties = self.get_properties()

    def get_properties(self):
        properties = {}
        try:
            with open(self.fileName, 'r') as f:
                for line in f.readlines():
                    split = line.strip().split('=')
                    if len(split) < 2: continue
                    if split[1].isdecimal():
                        properties[split[0]] = int(split[1])
                    elif is_float(split[1]):
                        properties[split[0]] = float(split[1])
                    else:
                        properties[split[0]] = split[1]
                self.properties = properties
        except IOError:
            print('exception arise.')

        return properties

    def get_property(self, prop):
        return self.properties[prop]
