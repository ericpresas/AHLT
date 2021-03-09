from collections import namedtuple
from configparser import ConfigParser


class config(object):
    def __init__(self, path='config/conf.env'):
        self.parser = ConfigParser()
        self.parser.read(path)

    def get(self, section):
        dict1 = {}
        options = self.parser.options(section)
        for option in options:
            try:
                dict1[option] = self.parser.get(section, option)
                if dict1[option] == -1:
                    print("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return namedtuple('Struct', dict1.keys())(*dict1.values())


config_file = config()