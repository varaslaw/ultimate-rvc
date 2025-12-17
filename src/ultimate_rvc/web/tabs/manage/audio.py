"""Module which defines the code for the "Audio" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.song_cover import get_named_song_dirs
from ultimate_rvc.core.manage.audio import (
    delete_all_audio,
    delete_all_dataset_audio,
    delete_all_intermediate_audio,
    delete_all_output_audio,
    delete_all_speech_audio,
    delete_dataset_audio,
    delete_intermediate_audio,
    delete_output_audio,
    delete_speech_audio,
    get_audio_datasets,
    get_named_audio_datasets,
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.web.common import setup_delete_event, update_dropdowns
from ultimate_rvc.web.config.event import ManageAudioEventState

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import AudioManagementConfig, TotalConfig


def render(total_config: TotalConfig) -> None:
    """
    Render "Audio" tab.

    Parameters
    ----------
    total_config: TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.

    """
    tab_config = total_config.management.audio
    tab_config.dummy_checkbox.instantiate()
    event_state = ManageAudioEventState()

    with gr.Tab("Удалить"):
        _render_intermediate_accordion(tab_config, event_state)
        _render_speech_accordion(tab_config, event_state)
        _render_output_accordion(tab_config, event_state)
        _render_dataset_accordion(tab_config, event_state)
        _render_all_accordion(tab_config, event_state)

        _, _, all_audio_update = [
            click_event.success(
                partial(
                    update_dropdowns,
                    get_named_song_dirs,
                    3 + len(total_config.song.multi_step.song_dirs.all),
                    [],
                    [0],
                ),
                outputs=[
                    tab_config.intermediate.instance,
                    total_config.song.one_click.cached_song.instance,
                    total_config.song.multi_step.cached_song.instance,
                    *total_config.song.multi_step.song_dirs.all,
                ],
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_intermediate_click.instance,
                event_state.delete_all_intermediate_click.instance,
                event_state.delete_all_click.instance,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
                outputs=tab_config.speech.instance,
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_speech_click.instance,
                event_state.delete_all_speech_click.instance,
                all_audio_update,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
                outputs=tab_config.output.instance,
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_output_click.instance,
                event_state.delete_all_output_click.instance,
                all_audio_update,
            ]
        ]

        for click_event in [
            event_state.delete_dataset_click.instance,
            event_state.delete_all_dataset_click.instance,
            all_audio_update,
        ]:
            click_event.success(
                partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
                outputs=tab_config.dataset.instance,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_audio_datasets, 1, [], [0]),
                outputs=total_config.training.multi_step.dataset.instance,
                show_progress="hidden",
            )


def _render_intermediate_accordion(
    tab_config: AudioManagementConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Промежуточное аудио", open=False), gr.Row():
        with gr.Column():
            tab_config.intermediate.instance.render()
            intermediate_audio_btn = gr.Button("Удалить выбранные", variant="secondary")
            all_intermediate_audio_btn = gr.Button("Удалить всё", variant="primary")
        with gr.Column():
            intermediate_audio_msg = gr.Textbox(
                label="Сообщение",
                interactive=False,
            )
        event_state.delete_intermediate_click.instance = setup_delete_event(
            intermediate_audio_btn,
            delete_intermediate_audio,
            [tab_config.dummy_checkbox.instance, tab_config.intermediate.instance],
            intermediate_audio_msg,
            "Удалить выбранные директории песен?",
            "[-] Выбранные директории удалены!",
        )
        event_state.delete_all_intermediate_click.instance = setup_delete_event(
            all_intermediate_audio_btn,
            delete_all_intermediate_audio,
            [tab_config.dummy_checkbox.instance],
            intermediate_audio_msg,
            "Удалить все промежуточные аудиофайлы?",
            "[-] Все промежуточные аудиофайлы удалены!",
        )


def _render_speech_accordion(
    tab_config: AudioManagementConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Аудио речи", open=False), gr.Row():
        with gr.Column():
            tab_config.speech.instance.render()
            speech_audio_btn = gr.Button("Удалить выбранные", variant="secondary")
            all_speech_audio_btn = gr.Button("Удалить всё", variant="primary")
        with gr.Column():
            speech_audio_msg = gr.Textbox(label="Сообщение", interactive=False)

        event_state.delete_speech_click.instance = setup_delete_event(
            speech_audio_btn,
            delete_speech_audio,
            [tab_config.dummy_checkbox.instance, tab_config.speech.instance],
            speech_audio_msg,
            "Удалить выбранные аудио с речью?",
            "[-] Выбранные аудио с речью удалены!",
        )

        event_state.delete_all_speech_click.instance = setup_delete_event(
            all_speech_audio_btn,
            delete_all_speech_audio,
            [tab_config.dummy_checkbox.instance],
            speech_audio_msg,
            "Удалить все аудио с речью?",
            "[-] Все аудио с речью удалены!",
        )


def _render_output_accordion(
    tab_config: AudioManagementConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Готовые аудио", open=False), gr.Row():
        with gr.Column():
            tab_config.output.instance.render()
            output_audio_btn = gr.Button("Удалить выбранные", variant="secondary")
            all_output_audio_btn = gr.Button("Удалить всё", variant="primary")
        with gr.Column():
            output_audio_msg = gr.Textbox(label="Сообщение", interactive=False)
        event_state.delete_output_click.instance = setup_delete_event(
            output_audio_btn,
            delete_output_audio,
            [tab_config.dummy_checkbox.instance, tab_config.output.instance],
            output_audio_msg,
            "Удалить выбранные готовые аудиофайлы?",
            "[-] Выбранные готовые файлы удалены!",
        )
        event_state.delete_all_output_click.instance = setup_delete_event(
            all_output_audio_btn,
            delete_all_output_audio,
            [tab_config.dummy_checkbox.instance],
            output_audio_msg,
            "Удалить все готовые аудиофайлы?",
            "[-] Все готовые аудиофайлы удалены!",
        )


def _render_dataset_accordion(
    tab_config: AudioManagementConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Аудио датасетов", open=False), gr.Row():
        with gr.Column():
            tab_config.dataset.instance.render()
            dataset_audio_btn = gr.Button("Удалить выбранные", variant="secondary")
            all_dataset_audio_btn = gr.Button("Удалить всё", variant="primary")
        with gr.Column():
            dataset_audio_msg = gr.Textbox(label="Сообщение", interactive=False)

        event_state.delete_dataset_click.instance = setup_delete_event(
            dataset_audio_btn,
            delete_dataset_audio,
            [tab_config.dummy_checkbox.instance, tab_config.dataset.instance],
            dataset_audio_msg,
            "Удалить выбранные аудио из датасетов?",
            "[-] Выбранные файлы из датасетов удалены!",
        )
        event_state.delete_all_dataset_click.instance = setup_delete_event(
            all_dataset_audio_btn,
            delete_all_dataset_audio,
            [tab_config.dummy_checkbox.instance],
            dataset_audio_msg,
            "Удалить все аудио из датасетов?",
            "[-] Все аудио из датасетов удалены!",
        )


def _render_all_accordion(
    tab_config: AudioManagementConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Всё аудио", open=True), gr.Row(equal_height=True):
        all_audio_btn = gr.Button("Удалить", variant="primary")
        all_audio_msg = gr.Textbox(label="Сообщение", interactive=False)

    event_state.delete_all_click.instance = setup_delete_event(
        all_audio_btn,
        delete_all_audio,
        [tab_config.dummy_checkbox.instance],
        all_audio_msg,
        "Удалить все аудиофайлы?",
        "[-] Все аудиофайлы удалены!",
    )
