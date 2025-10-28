"""Module artefacts.py"""
import os

import pandas as pd
import sklearn
import tensorflow as tf

import src.functions.streams


class Artefacts:
    """
    Artefacts
    """

    def __init__(self, model: tf.keras.src.models.Sequential, scaler: sklearn.preprocessing.MinMaxScaler, arguments: dict, path: str):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__model = model
        self.__scaler = scaler
        self.__arguments = arguments
        self.__path = path

        # Instances
        self.__streams = src.functions.streams.Streams()

    def __history(self):
        """

        :return:
        """

        history = pd.DataFrame(data=self.__model.history.history)

        return self.__streams.write(blob=history, path=os.path.join(self.__path, 'history.csv'))

    def exc(self):

        self.__history()
