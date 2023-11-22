import numpy as np

class RepeatEnv:
    def __init__(self, strings):
        self._strings = strings

    def reset(self):
        _id = np.random.randint(0, len(self._strings))
        self.__current_objective = self._strings[_id]
        return f"Say {self.__current_objective}\nYour turn:", None

    def step(self, string):
        r = 1 if string == self.__current_objective else 0
        return "", r, r == 1, None