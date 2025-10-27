import pandas as pd
import numpy as np

import sklearn


class Scaling:
    """
    Minimum Maximum Scaling
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__features = arguments.get('scaling').get('features')

    def __restructure(self, structure: pd.DataFrame, transforms: np.ndarray):
        """

        :param structure: The dataframe the transforms apply to
        :param transforms: The scaled forms of the fields of self.__features
        :return:
        """

        __data = structure.drop(columns=self.__features)
        __data.loc[:, self.__features] = transforms

        return __data

    def reference(self, blob: pd.DataFrame):
        """

        :param blob: The training data
        :return:
            frame: A data frame wherein the features in focus have been scaled.
            scaler: The scaler object vis-à-vis the training data.
        """

        scaler = sklearn.preprocessing.MinMaxScaler()

        # Creating and using a scaler
        transforms = scaler.fit_transform(blob.copy()[self.__features])
        frame = self.__restructure(structure=blob, transforms=transforms)

        return frame, scaler

    def exc(self, blob: pd.DataFrame, scaler: sklearn.preprocessing.MinMaxScaler):
        """
        For transforming data sets associated with the training data that built the scaler

        :param blob:
        :param scaler: A scaler object
        """

        transforms = scaler.transform(blob.copy()[self.__features])

        return self.__restructure(structure=blob, transforms=transforms)
