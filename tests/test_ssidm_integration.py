import unittest
from types import SimpleNamespace

import torch

from pidm_imitation.agents.models.ssidm import SSIDMPolicyNetwork
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_valid_agents import (
    ValidPytorchAgents,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.inference_models import (
    select_rollout_action,
)
from pidm_imitation.agents.supervised_learning.inputs_factory import InputsFactory
from pidm_imitation.agents.supervised_learning.model_factory import ModelFactory
from pidm_imitation.constants import STATE_HISTORY_KEY, STATE_LOOKAHEAD_KEY
from pidm_imitation.evaluation.toy_valid_agents import ValidToyAgents
from pidm_imitation.evaluation.toy_utils import create_toy_config_parser
from pidm_imitation.utils.valid_models import ValidModels


class SSIDMIntegrationTests(unittest.TestCase):
    def _get_config_and_model(self, config_path: str, agent_name: str):
        config = create_toy_config_parser(config_path, agent_name)
        datamodule = SimpleNamespace(
            state_dim=4,
            history=0,
            lookahead=1,
            include_k=False,
            lookahead_slice=[0],
        )
        model = ModelFactory.get_model(config, datamodule)
        return config, model

    def test_registries_include_new_algorithms(self):
        self.assertIn(ValidModels.PSSIDM, ValidModels.ALL)
        self.assertIn(ValidModels.LSSIDM, ValidModels.ALL)
        self.assertIn(ValidPytorchAgents.PSSIDM, ValidPytorchAgents.ALL)
        self.assertIn(ValidPytorchAgents.LSSIDM, ValidPytorchAgents.ALL)
        self.assertIn(ValidToyAgents.TOY_PSSIDM, ValidToyAgents.ALL)
        self.assertIn(ValidToyAgents.TOY_LSSIDM, ValidToyAgents.ALL)

    def test_input_routing_is_state_history_plus_lookahead_only(self):
        for config_path, agent_name in [
            ("configs/supervised_learning/pssidm_example.yaml", "toy_pssidm"),
            ("configs/supervised_learning/lssidm_example.yaml", "toy_lssidm"),
        ]:
            config = create_toy_config_parser(config_path, agent_name)
            inputs = InputsFactory.get_inputs(config)
            self.assertEqual(
                inputs["policy_head"][1],
                [STATE_HISTORY_KEY, STATE_LOOKAHEAD_KEY],
            )

    def test_ssidm_models_use_shared_recurrent_policy_impl(self):
        _, pssidm_model = self._get_config_and_model(
            "configs/supervised_learning/pssidm_example.yaml", "toy_pssidm"
        )
        _, lssidm_model = self._get_config_and_model(
            "configs/supervised_learning/lssidm_example.yaml", "toy_lssidm"
        )

        self.assertTrue(pssidm_model.is_recurrent)
        self.assertTrue(lssidm_model.is_recurrent)
        self.assertIsInstance(pssidm_model.policy_head.policy_model, SSIDMPolicyNetwork)
        self.assertIsInstance(lssidm_model.policy_head.policy_model, SSIDMPolicyNetwork)
        self.assertFalse(pssidm_model.policy_head.policy_model.use_latent_encoder)
        self.assertTrue(lssidm_model.policy_head.policy_model.use_latent_encoder)

    def test_rollout_action_selection_uses_last_sequence_step(self):
        predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        selected = select_rollout_action(predicted)
        self.assertTrue(torch.equal(selected, torch.tensor([[5.0, 6.0]])))

    def test_ssidm_requires_fixed_next_state_lookahead(self):
        config = create_toy_config_parser(
            "configs/supervised_learning/pssidm_example.yaml", "toy_pssidm"
        )
        bad_datamodule = SimpleNamespace(
            state_dim=4,
            history=0,
            lookahead=1,
            include_k=True,
            lookahead_slice=[0, 1],
        )
        with self.assertRaises(AssertionError):
            ModelFactory.get_model(config, bad_datamodule)


if __name__ == "__main__":
    unittest.main()
