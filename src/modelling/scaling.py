import pandas as pd
import numpy as np

import sklearn


class Scaling:

    def __init__(self, features__: list) -> None:
        """

        :param features__: The fields that would undergo scaling
        """

        self.__features = features__

    def __restructure(self, structure: pd.DataFrame, transforms: np.ndarray):
        """

        :param structure:
        :param transforms:
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

        :param blob:
        :param scaler:
        """

        transforms = scaler.transform(blob.copy()[self.__features])

        return self.__restructure(structure=blob, transforms=transforms)
