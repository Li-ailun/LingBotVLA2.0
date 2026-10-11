"""RTC-enabled copy of the official LingBot FlowMatchingV2 sampling path.

This module owns no parameters and never replaces the official class globally.
"""

import torch

from .rtc_sampler import apply_rtc_velocity_guidance


def denoise_actions_rtc(flow_model, state, prefix_pad_masks, past_key_values, noise, *, prefix_position_ids=None, rtc_request=None):
    """Run the official Euler loop, inserting RTC after velocity prediction."""
    dt = torch.tensor(-1.0 / flow_model.config.num_steps, dtype=noise.dtype, device=noise.device)
    x_t = noise
    time = torch.tensor(1.0, dtype=noise.dtype, device=noise.device)
    predict = flow_model.predict_velocity
    if getattr(flow_model, "_use_compile_predict_velocity", False) and rtc_request is None:
        predict = getattr(flow_model, "_compiled_predict_velocity", None) or torch.compile(
            flow_model.predict_velocity, fullgraph=False, dynamic=False,
            options={"triton.cudagraphs": False},
        )
        flow_model._compiled_predict_velocity = predict
    while time >= -dt / 2:
        expanded_time = time.expand(x_t.shape[0])

        def base_velocity(value):
            return predict(state, prefix_pad_masks, past_key_values, value, expanded_time,
                           prefix_position_ids=prefix_position_ids)

        velocity = apply_rtc_velocity_guidance(base_velocity, x_t, time, rtc_request)
        x_t += dt * velocity
        time += dt
    return x_t


def sample_actions_rtc(flow_model, images, img_masks, lang_tokens, lang_masks, state, *, rtc_request=None, noise=None, image_grid_thw=None):
    """Official prefix/cache setup followed by the isolated RTC denoiser."""
    if noise is None:
        noise = torch.randn(
            (state.shape[0], flow_model.config.n_action_steps, flow_model.config.max_action_dim),
            device=state.device, dtype=state.dtype,
        )
    (prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids,
     visual_pos_masks, deepstack_visual_embeds) = flow_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, image_grid_thw=image_grid_thw,
    )
    # Imported lazily so CPU math tests do not require the complete LingBot stack.
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    _, cache, _ = flow_model.qwenvl_with_expert.forward(
        attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids,
        vlm_position_ids=prefix_position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=flow_model.config.use_cache,
        fill_kv_cache=True, visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    return denoise_actions_rtc(
        flow_model, state, prefix_pad_masks, cache, noise,
        prefix_position_ids=prefix_position_ids, rtc_request=rtc_request,
    )


def install_rtc_sampler(policy):
    """Attach a local entrypoint without changing weights or state-dict keys."""
    model = policy.model if hasattr(policy, "model") else policy

    def bound(*args, rtc_request=None, **kwargs):
        return sample_actions_rtc(model, *args, rtc_request=rtc_request, **kwargs)

    model.sample_actions_rtc = bound
