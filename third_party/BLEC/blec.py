class BLECMetrics(object):
    def __init__(self, language='Default'):
        self.language = language
    # Returns the list of tokens that are not matched

    def evaluate(self, pred, logic, gold) -> list:
        raise NotImplementedError

