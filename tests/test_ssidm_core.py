import unittest

import torch

from pidm_imitation.agents.models.ssidm import SSIDMPolicyNetwork, StructuredSSMCore


class StructuredSSMCoreTests(unittest.TestCase):
    def test_recurrent_and_convolution_match_random_init(self):
        torch.manual_seed(0)
        core = StructuredSSMCore(
            stream_dim=3,
            action_dim=2,
            ssm_state_dim=5,
            delta_init=0.5,
            hippo_init=False,
            diagonal_A=True,
        )
        u = torch.randn(4, 16, 3)
        u_ref = torch.randn(4, 16, 3)

        recurrent = core.forward_recurrent(u, u_ref)
        convolution = core.forward_convolve(u, u_ref)

        self.assertLess((recurrent - convolution).abs().max().item(), 1e-5)

    def test_recurrent_and_convolution_match_hippo_init(self):
        torch.manual_seed(0)
        core = StructuredSSMCore(
            stream_dim=3,
            action_dim=2,
            ssm_state_dim=6,
            delta_init=0.5,
            hippo_init=True,
            diagonal_A=True,
        )
        u = torch.randn(4, 16, 3)
        u_ref = torch.randn(4, 16, 3)

        recurrent = core.forward_recurrent(u, u_ref)
        convolution = core.forward_convolve(u, u_ref)

        self.assertLess((recurrent - convolution).abs().max().item(), 1e-5)

    def test_lag_zero_kernel_term(self):
        torch.manual_seed(0)
        core = StructuredSSMCore(
            stream_dim=2,
            action_dim=2,
            ssm_state_dim=4,
            delta_init=0.25,
            hippo_init=True,
            diagonal_A=True,
        )
        _, B_bar = core.discretise()
        D = core.get_D()
        kernel = core.build_kernel(seq_len=8)

        expected_k0 = core.C @ B_bar + D
        self.assertTrue(torch.allclose(kernel[0], expected_k0, atol=1e-6, rtol=1e-6))

    def test_continuous_A_is_stable(self):
        core = StructuredSSMCore(
            stream_dim=2,
            action_dim=2,
            ssm_state_dim=8,
            delta_init=1.0,
            hippo_init=True,
            diagonal_A=True,
        )
        eigvals = torch.linalg.eigvals(core.get_A())
        self.assertTrue(torch.all(eigvals.real < 0))


class SSIDMPolicyNetworkTests(unittest.TestCase):
    def test_eval_step_matches_recurrent_sequence_for_pssidm_and_lssidm(self):
        torch.manual_seed(0)
        inputs = torch.randn(1, 6, 8)

        for use_latent_encoder in [False, True]:
            model = SSIDMPolicyNetwork(
                input_dim=8,
                state_dim=4,
                action_type="left_stick",
                d_model=6,
                num_ssm_layers=3,
                ssm_state_dim=6,
                delta_init=0.5,
                hippo_init=True,
                diagonal_A=True,
                latent_encoder_dim=5,
                use_latent_encoder=use_latent_encoder,
            )
            model.eval()

            model.reset()
            rollout_outputs = []
            for k in range(inputs.shape[1]):
                rollout_outputs.append(model(inputs[:, k : k + 1, :]))
            rollout_outputs = torch.cat(rollout_outputs, dim=1)

            model.reset()
            recurrent_outputs = model.forward_recurrent(inputs)

            self.assertTrue(
                torch.allclose(
                    rollout_outputs,
                    recurrent_outputs,
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

    def test_training_convolution_matches_recurrent_sequence_for_stacked_pssidm(self):
        torch.manual_seed(0)
        model = SSIDMPolicyNetwork(
            input_dim=8,
            state_dim=4,
            action_type="left_stick",
            d_model=7,
            num_ssm_layers=4,
            ssm_state_dim=5,
            delta_init=0.5,
            hippo_init=True,
            diagonal_A=True,
            use_latent_encoder=False,
        )
        inputs = torch.randn(2, 9, 8)

        convolution = model.forward_convolution(inputs)
        recurrent = model.forward_recurrent(inputs)

        self.assertTrue(
            torch.allclose(
                convolution,
                recurrent,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_training_convolution_matches_recurrent_sequence_for_nonlinear_variants(self):
        torch.manual_seed(0)
        inputs = torch.randn(2, 9, 8)

        for block_nonlinearity, prenorm in [
            ("none", False),
            ("silu", False),
            ("silu", True),
            ("gelu", False),
        ]:
            model = SSIDMPolicyNetwork(
                input_dim=8,
                state_dim=4,
                action_type="left_stick",
                d_model=7,
                num_ssm_layers=3,
                ssm_state_dim=5,
                delta_init=0.5,
                hippo_init=True,
                diagonal_A=True,
                block_nonlinearity=block_nonlinearity,
                prenorm=prenorm,
                dropout=0.0,
                use_latent_encoder=False,
            )

            convolution = model.forward_convolution(inputs)
            recurrent = model.forward_recurrent(inputs)

            self.assertTrue(
                torch.allclose(
                    convolution,
                    recurrent,
                    atol=1e-5,
                    rtol=1e-5,
                ),
                msg=f"Mismatch for block_nonlinearity={block_nonlinearity}, prenorm={prenorm}",
            )

    def test_eval_step_matches_recurrent_sequence_for_nonlinear_lssidm(self):
        torch.manual_seed(0)
        inputs = torch.randn(1, 6, 8)
        model = SSIDMPolicyNetwork(
            input_dim=8,
            state_dim=4,
            action_type="left_stick",
            d_model=6,
            num_ssm_layers=3,
            ssm_state_dim=6,
            delta_init=0.5,
            hippo_init=True,
            diagonal_A=True,
            block_nonlinearity="silu",
            prenorm=True,
            latent_encoder_dim=6,
            use_latent_encoder=True,
        )
        model.eval()

        model.reset()
        rollout_outputs = []
        for k in range(inputs.shape[1]):
            rollout_outputs.append(model(inputs[:, k : k + 1, :]))
        rollout_outputs = torch.cat(rollout_outputs, dim=1)

        model.reset()
        recurrent_outputs = model.forward_recurrent(inputs)

        self.assertTrue(
            torch.allclose(
                rollout_outputs,
                recurrent_outputs,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_invalid_block_nonlinearity_raises(self):
        with self.assertRaises(ValueError):
            SSIDMPolicyNetwork(
                input_dim=8,
                state_dim=4,
                action_type="left_stick",
                d_model=6,
                num_ssm_layers=2,
                ssm_state_dim=4,
                block_nonlinearity="relu",
                use_latent_encoder=False,
            )

    def test_reset_clears_all_block_caches(self):
        torch.manual_seed(0)
        model = SSIDMPolicyNetwork(
            input_dim=8,
            state_dim=4,
            action_type="left_stick",
            d_model=6,
            num_ssm_layers=3,
            ssm_state_dim=4,
            delta_init=0.5,
            hippo_init=True,
            diagonal_A=True,
            use_latent_encoder=False,
        )
        model.eval()
        inputs = torch.randn(1, 3, 8)

        for k in range(inputs.shape[1]):
            model(inputs[:, k : k + 1, :])

        self.assertTrue(all(block._cached_state is not None for block in model.blocks))

        model.reset()

        self.assertTrue(all(block._cached_state is None for block in model.blocks))


if __name__ == "__main__":
    unittest.main()
