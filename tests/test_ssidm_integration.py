import unittest
from types import SimpleNamespace

import torch

from pidm_imitation.agents.models.ssidm import SSIDMPolicyNetwork
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_agents import (
    PytorchIdmAgent,
)
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_valid_agents import (
    ValidPytorchAgents,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.observation_handlers import (
    StateHandler,
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
        self.assertEqual(pssidm_model.policy_head.policy_model.d_model, 64)
        self.assertEqual(lssidm_model.policy_head.policy_model.d_model, 64)
        self.assertEqual(len(pssidm_model.policy_head.policy_model.blocks), 3)
        self.assertEqual(len(lssidm_model.policy_head.policy_model.blocks), 3)
        self.assertEqual(
            pssidm_model.policy_head.policy_model.block_nonlinearity, "none"
        )
        self.assertEqual(
            lssidm_model.policy_head.policy_model.block_nonlinearity, "none"
        )
        self.assertFalse(pssidm_model.policy_head.policy_model.prenorm)
        self.assertFalse(lssidm_model.policy_head.policy_model.prenorm)

    def test_nonlinear_smoke_configs_instantiate(self):
        for config_path, agent_name, expected_nonlinearity, expected_prenorm, expected_latent in [
            (
                "configs/supervised_learning/pssidm_silu_smoke.yaml",
                "toy_pssidm",
                "silu",
                False,
                False,
            ),
            (
                "configs/supervised_learning/pssidm_silu_prenorm_smoke.yaml",
                "toy_pssidm",
                "silu",
                True,
                False,
            ),
            (
                "configs/supervised_learning/lssidm_silu_smoke.yaml",
                "toy_lssidm",
                "silu",
                False,
                True,
            ),
            (
                "configs/supervised_learning/lssidm_silu_prenorm_smoke.yaml",
                "toy_lssidm",
                "silu",
                True,
                True,
            ),
        ]:
            _, model = self._get_config_and_model(config_path, agent_name)
            policy_model = model.policy_head.policy_model
            self.assertIsInstance(policy_model, SSIDMPolicyNetwork)
            self.assertEqual(policy_model.block_nonlinearity, expected_nonlinearity)
            self.assertEqual(policy_model.prenorm, expected_prenorm)
            self.assertEqual(policy_model.use_latent_encoder, expected_latent)

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

    def test_recurrent_idm_agent_passes_flat_state_to_planner(self):
        config = create_toy_config_parser(
            "configs/supervised_learning/pssidm_example.yaml", "toy_pssidm"
        )
        datamodule = SimpleNamespace(
            state_dim=4,
            history=0,
            lookahead=1,
            include_k=False,
            lookahead_slice=[0],
        )
        model = ModelFactory.get_model(config, datamodule)

        class DummyPlanner:
            def __init__(self):
                self.current_state_shape = None

            def get_lookahead_state_action_and_k(
                self, current_state, current_action, current_step
            ):
                self.current_state_shape = tuple(current_state.shape)
                return (
                    torch.zeros(4, dtype=torch.float32),
                    torch.zeros(2, dtype=torch.float32),
                    0,
                )

        planner = DummyPlanner()
        agent = PytorchIdmAgent(
            config=config,
            model=model,
            model_hparams={},
            data_hparams={
                "history": 0,
                "history_slice": [],
                "padding_strategy": "zero",
                "lookahead": 1,
                "include_k": False,
                "lookahead_slice": [0],
            },
            model_path=".",
            observation_handler=StateHandler(config.state_config.type),
            idm_planner=planner,  # type: ignore[arg-type]
        )
        agent._get_inputs(
            raw_obs=None,
            built_obs={"states": [1.0, 2.0, 3.0, 4.0]},
        )
        self.assertEqual(planner.current_state_shape, (4,))


if __name__ == "__main__":
    unittest.main()
