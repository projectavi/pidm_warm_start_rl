# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class ValidModels:
    BC = "bc"
    IDM = "idm"
    PSSIDM = "pssidm"
    LSSIDM = "lssidm"
    ALL = [BC, IDM, PSSIDM, LSSIDM]

    @staticmethod
    def is_valid_model(model_name: str) -> bool:
        """
        Check if the given model name is a valid model.
        :param model_name: The name of the model to check.
        :return: True if the model name is valid, False otherwise.
        """
        return model_name in ValidModels.ALL
