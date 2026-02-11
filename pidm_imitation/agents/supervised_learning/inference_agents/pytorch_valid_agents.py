# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class ValidPytorchAgents:
    IDM = "idm"
    BC = "bc"
    ALL = [IDM, BC]

    @staticmethod
    def is_pytorch_agent(agent_name: str) -> bool:
        return agent_name in ValidPytorchAgents.ALL
