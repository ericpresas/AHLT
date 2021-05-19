class Utils(object):

    @staticmethod
    def start_end_between(pos1, pos2):
        start = min([pos1[1], pos2[1]])
        end = max([pos1[0], pos2[0]])
        return [start, end]

    @staticmethod
    def before_position(pos1, pos2):
        return min([pos1[0], pos2[0]])

    @staticmethod
    def after_position(pos1, pos2):
        return max([pos1[1], pos2[1]])

    @staticmethod
    def distance_between(pos1, pos2):
        start = min([pos1[1], pos2[1]])
        end = max([pos1[0], pos2[0]])
        return end - start

    @staticmethod
    def is_special_char(token):
        special_chars = "!@#$%^&*()-+?_=,<>/-[]:.;"
        return token in special_chars

    @staticmethod
    def is_numeric(token):
        try:
            float(token)
            return True
        except ValueError:
            return token.isnumeric()