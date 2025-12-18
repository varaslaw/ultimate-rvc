"""
Module which defines the code for the "Generate speech - multi-step
generation" tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.common import SPEECH_DIR
from ultimate_rvc.core.generate.common import convert
from ultimate_rvc.core.generate.speech import (
    get_mixed_speech_track_name,
    mix_speech,
    run_edge_tts,
)
from ultimate_rvc.core.manage.audio import (
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.typing_extra import EmbedderModel, RVCContentType
from ultimate_rvc.web.common import (
    exception_harness,
    render_transfer_component,
    setup_transfer_event,
    toggle_visibility,
    toggle_visible_component,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SpeechTransferOption

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import TotalConfig


def render(total_config: TotalConfig) -> None:
    """
    Render "Generate speech - multi-step generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.

    """
    tab_config = total_config.speech.multi_step
    for c in tab_config.input_audio.all:
        c.instantiate()
    with gr.Tab("По шагам"):
        _render_step_1(total_config)
        _render_step_2(total_config)
        _render_step_3(total_config)


def _render_step_1(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step
    with gr.Accordion("Шаг 1: текст → речь", open=True):
        with gr.Row():
            with gr.Column():
                tab_config.source_type.instantiate()
            with gr.Column():
                tab_config.source.instantiate()
                local_file = gr.File(
                    label="Источник",
                    file_types=[".txt"],
                    file_count="single",
                    type="filepath",
                    visible=False,
                )
            tab_config.source_type.instance.input(
                partial(toggle_visible_component, 2),
                inputs=tab_config.source_type.instance,
                outputs=[tab_config.source.instance, local_file],
                show_progress="hidden",
            )
            local_file.change(
                update_value,
                inputs=local_file,
                outputs=tab_config.source.instance,
                show_progress="hidden",
            )
        tab_config.edge_tts_voice.instance.render()
        with gr.Accordion("Настройки", open=False):
            with gr.Row():
                tab_config.tts_pitch_shift.instantiate()
                tab_config.tts_speed_change.instantiate()
                tab_config.tts_volume_change.instantiate()
            speech_transfer = _render_speech_transfer(
                [SpeechTransferOption.STEP_2_SPEECH],
                "Речь",
            )
        with gr.Row():
            tts_reset_btn = gr.Button("Сбросить настройки")
            tts_btn = gr.Button("Преобразовать текст", variant="primary")
        tts_transfer_btn = gr.Button("Передать речь")

        speech_track_output = gr.Audio(
            label="Сгенерированная речь",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )
        tts_reset_btn.click(
            lambda: [
                tab_config.tts_pitch_shift.value,
                tab_config.tts_speed_change.value,
                tab_config.tts_volume_change.value,
                gr.Dropdown(value=[SpeechTransferOption.STEP_2_SPEECH]),
            ],
            outputs=[
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
                speech_transfer,
            ],
            show_progress="hidden",
        )
        tts_btn.click(
            exception_harness(run_edge_tts, info_msg="Текст успешно озвучен!"),
            inputs=[
                tab_config.source.instance,
                tab_config.edge_tts_voice.instance,
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
            ],
            outputs=speech_track_output,
        ).then(
            partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
            outputs=total_config.management.audio.speech.instance,
            show_progress="hidden",
        )
        setup_transfer_event(
            tts_transfer_btn,
            speech_transfer,
            speech_track_output,
            tab_config.input_audio.all,
        )


def _render_step_2(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step

    with gr.Accordion("Шаг 2: конверсия речи", open=False):
        tab_config.input_audio.speech.instance.render()
        tab_config.voice_model.instance.render()
        with gr.Accordion("Настройки", open=False):
            with gr.Row():
                tab_config.n_octaves.instantiate()
                tab_config.n_semitones.instantiate()
            with gr.Accordion("Настройки синтеза", open=False):
                with gr.Row():
                    tab_config.f0_method.instantiate()
                    tab_config.index_rate.instantiate()
                with gr.Row():
                    tab_config.rms_mix_rate.instantiate()
                    tab_config.protect_rate.instantiate()
            with gr.Accordion("Обогащение речи", open=False):
                with gr.Row(), gr.Column():
                    tab_config.split_voice.instantiate()
                with gr.Row():
                    with gr.Column():
                        tab_config.autotune_voice.instantiate()
                        tab_config.autotune_strength.instantiate()
                    with gr.Column():
                        tab_config.proposed_pitch.instantiate()
                        tab_config.proposed_pitch_threshold.instantiate()
                    with gr.Column():
                        tab_config.clean_voice.instantiate()
                        tab_config.clean_strength.instantiate()
            tab_config.autotune_voice.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.autotune_voice.instance,
                outputs=tab_config.autotune_strength.instance,
                show_progress="hidden",
            )
            tab_config.proposed_pitch.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.proposed_pitch.instance,
                outputs=tab_config.proposed_pitch_threshold.instance,
                show_progress="hidden",
            )
            tab_config.clean_voice.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.clean_voice.instance,
                outputs=tab_config.clean_strength.instance,
                show_progress="hidden",
            )
            with gr.Accordion("Speaker embedding settings", open=False), gr.Row():
                with gr.Column():
                    tab_config.embedder_model.instantiate()
                    tab_config.custom_embedder_model.instance.render()
                tab_config.sid.instantiate()
            tab_config.embedder_model.instance.change(
                partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                inputs=tab_config.embedder_model.instance,
                outputs=tab_config.custom_embedder_model.instance,
                show_progress="hidden",
            )
            converted_speech_transfer = _render_speech_transfer(
                [SpeechTransferOption.STEP_3_SPEECH],
                "Сконвертированная речь",
            )
        with gr.Row():
            convert_speech_reset_btn = gr.Button("Сбросить настройки")
            convert_speech_btn = gr.Button("Конвертировать речь", variant="primary")
        converted_speech_transfer_btn = gr.Button("Передать сконвертированную речь")

        converted_speech_track_output = gr.Audio(
            label="Сконвертированная речь",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )
        convert_speech_reset_btn.click(
            lambda: [
                tab_config.n_octaves.value,
                tab_config.n_semitones.value,
                tab_config.f0_method.value,
                tab_config.index_rate.value,
                tab_config.rms_mix_rate.value,
                tab_config.protect_rate.value,
                tab_config.split_voice.value,
                tab_config.autotune_voice.value,
                tab_config.autotune_strength.value,
                tab_config.proposed_pitch.value,
                tab_config.proposed_pitch_threshold.value,
                tab_config.clean_voice.value,
                tab_config.clean_strength.value,
                tab_config.embedder_model.value,
                tab_config.sid.value,
                gr.Dropdown(value=[SpeechTransferOption.STEP_3_SPEECH]),
            ],
            outputs=[
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_method.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.proposed_pitch.instance,
                tab_config.proposed_pitch_threshold.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.sid.instance,
                converted_speech_transfer,
            ],
            show_progress="hidden",
        )

        convert_speech_btn.click(
            partial(
                exception_harness(convert, info_msg="Речь успешно сконвертирована!"),
                content_type=RVCContentType.SPEECH,
                make_directory=True,
            ),
            inputs=[
                tab_config.input_audio.speech.instance,
                gr.State(SPEECH_DIR),
                tab_config.voice_model.instance,
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_method.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.proposed_pitch.instance,
                tab_config.proposed_pitch_threshold.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.custom_embedder_model.instance,
                tab_config.sid.instance,
            ],
            outputs=converted_speech_track_output,
            concurrency_id=ConcurrencyId.GPU,
            concurrency_limit=1,
        ).then(
            partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
            outputs=total_config.management.audio.speech.instance,
            show_progress="hidden",
        )
        setup_transfer_event(
            converted_speech_transfer_btn,
            converted_speech_transfer,
            converted_speech_track_output,
            tab_config.input_audio.all,
        )


def _render_step_3(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step
    with gr.Accordion("Шаг 3: сведение речи", open=False):
        tab_config.input_audio.converted_speech.instance.render()
        with gr.Accordion("Настройки", open=False):
            with gr.Row():
                tab_config.output_gain.instantiate()
                tab_config.output_sr.instantiate()
            with gr.Row():
                tab_config.output_name.instantiate(
                    value=partial(
                        update_output_name,
                        get_mixed_speech_track_name,
                        False,  # noqa: FBT003
                    ),
                    inputs=[
                        gr.State(None),
                        gr.State(None),
                        tab_config.input_audio.converted_speech.instance,
                    ],
                )
                tab_config.output_format.instantiate()
            mixed_speech_transfer = _render_speech_transfer([], "Сведённая речь")
        with gr.Row():
            mix_speech_btn = gr.Button("Свести речь", variant="primary")
            mix_speech_transfer_btn = gr.Button("Передать сведённую речь")
        mix_speech_reset_btn = gr.Button("Сбросить настройки")
        mixed_speech_track_output = gr.Audio(
            label="Сведённая речь",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

        mix_speech_reset_btn.click(
            lambda: [
                tab_config.output_gain.value,
                tab_config.output_sr.value,
                tab_config.output_format.value,
                gr.Dropdown(value=[]),
            ],
            outputs=[
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                mixed_speech_transfer,
            ],
            show_progress="hidden",
        )
        mix_speech_btn.click(
            exception_harness(mix_speech, info_msg="Речь успешно сведена!"),
            inputs=[
                tab_config.input_audio.converted_speech.instance,
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.output_name.instance,
            ],
            outputs=mixed_speech_track_output,
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
            outputs=total_config.management.audio.output.instance,
            show_progress="hidden",
        )
        setup_transfer_event(
            mix_speech_transfer_btn,
            mixed_speech_transfer,
            mixed_speech_track_output,
            tab_config.input_audio.all,
        )


def _render_speech_transfer(
    value: list[SpeechTransferOption],
    label_prefix: str,
) -> gr.Dropdown:
    return render_transfer_component(value, label_prefix, SpeechTransferOption)
