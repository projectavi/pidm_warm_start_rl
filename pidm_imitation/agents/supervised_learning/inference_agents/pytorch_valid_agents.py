# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class ValidPytorchAgents:
    IDM = "idm"
    BC = "bc"
    PSSIDM = "pssidm"
    LSSIDM = "lssidm"
    ALL = [IDM, BC, PSSIDM, LSSIDM]

    @staticmethod
    def is_pytorch_agent(agent_name: str) -> bool:
        return agent_name in ValidPytorchAgents.ALL
