# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class ValidToyAgents:
    TOY_IDM = "toy_idm"
    TOY_BC = "toy_bc"
    TOY_PSSIDM = "toy_pssidm"
    TOY_LSSIDM = "toy_lssidm"
    ALL = [
        TOY_BC,
        TOY_IDM,
        TOY_PSSIDM,
        TOY_LSSIDM,
    ]

    @staticmethod
    def is_toy_agent(agent_name: str) -> bool:
        return agent_name in ValidToyAgents.ALL
