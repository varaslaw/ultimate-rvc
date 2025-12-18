"""Module which defines the code for the "Settings" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.manage.config import (
    delete_all_configs,
    delete_configs,
    get_config_names,
)
from ultimate_rvc.core.manage.settings import delete_temp_files
from ultimate_rvc.web.common import (
    confirm_box_js,
    confirmation_harness,
    exception_harness,
    load_total_config_values,
    render_msg,
    save_total_config_values,
    setup_delete_event,
    update_dropdowns,
)

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import SettingsManagementConfig, TotalConfig


def render(total_config: TotalConfig) -> None:
    """
    Render "Settings" tab.

    Parameters
    ----------
    total_config : TotalConfig
        The total configuration object containing all the settings.

    """
    tab_config = total_config.management.settings
    tab_config.dummy_checkbox.instantiate()
    _render_config_files_tab(tab_config, total_config)
    _render_temp_files_tab(tab_config)


def _render_config_files_tab(
    tab_config: SettingsManagementConfig,
    total_config: TotalConfig,
) -> None:
    components = [config.instance for config in total_config.all]
    with gr.Tab("Файлы конфигурации"):
        with gr.Accordion("Сохранить конфигурацию"), gr.Row():
            with gr.Column():
                save_config_name = gr.Textbox(
                    label="Имя конфигурации",
                    info=(
                        "Название конфигурации, в которую будут сохранены текущие"
                        " настройки интерфейса."
                    ),
                    placeholder="Введите название конфигурации",
                    max_lines=1,
                    value="Конфигурация по умолчанию",
                )
                save_config_btn = gr.Button("Сохранить", variant="primary")
            with gr.Column():
                save_config_msg = gr.Textbox(label="Сообщение", interactive=False)

        with gr.Accordion("Загрузить конфигурацию", open=False), gr.Row():
            with gr.Column():
                tab_config.load_config_name.instance.render()
                load_config_btn = gr.Button("Загрузить", variant="primary")
            with gr.Column():
                load_config_msg = gr.Textbox(label="Сообщение", interactive=False)

        save_config_btn.click(
            exception_harness(save_total_config_values),
            inputs=[save_config_name, *components],
            outputs=save_config_msg,
        ).success(
            partial(render_msg, "[-] Конфигурация успешно сохранена!"),
            outputs=save_config_msg,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_config_names, 1, value_indices=[0]),
            inputs=save_config_name,
            outputs=tab_config.load_config_name.instance,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_config_names, 1, [], [0]),
            outputs=tab_config.delete_config_names.instance,
            show_progress="hidden",
        )

        load_config_btn.click(
            exception_harness(load_total_config_values),
            inputs=tab_config.load_config_name.instance,
            outputs=components,
            show_progress_on=load_config_msg,
        ).success(
            partial(render_msg, "[-] Конфигурация успешно загружена!"),
            outputs=load_config_msg,
            show_progress="hidden",
        )

        with gr.Accordion("Удалить конфигурации", open=False), gr.Row():
            with gr.Column():
                tab_config.delete_config_names.instance.render()
                delete_config_btn = gr.Button("Удалить выбранные", variant="secondary")
                delete_all_config_btn = gr.Button("Удалить все", variant="primary")
            with gr.Column():
                delete_config_msg = gr.Textbox(
                    label="Сообщение",
                    interactive=False,
                )
        delete_config_click = setup_delete_event(
            delete_config_btn,
            delete_configs,
            [
                tab_config.dummy_checkbox.instance,
                tab_config.delete_config_names.instance,
            ],
            delete_config_msg,
            "Удалить конфигурации с выбранными именами?",
            "[-] Выбранные конфигурации удалены!",
        )
        delete_all_config_click = setup_delete_event(
            delete_all_config_btn,
            delete_all_configs,
            [tab_config.dummy_checkbox.instance],
            delete_config_msg,
            "Удалить все конфигурации?",
            "[-] Все конфигурации удалены!",
        )
        for event in [delete_config_click, delete_all_config_click]:
            event.success(
                partial(update_dropdowns, get_config_names, 2, [], [1]),
                outputs=[
                    tab_config.load_config_name.instance,
                    tab_config.delete_config_names.instance,
                ],
                show_progress="hidden",
            )


def _render_temp_files_tab(tab_config: SettingsManagementConfig) -> None:
    with gr.Tab("Временные файлы"):
        gr.Markdown("")
        with gr.Row(equal_height=True):
            temporary_files_btn = gr.Button("Удалить всё", variant="primary")
            temporary_files_msg = gr.Textbox(label="Сообщение", interactive=False)

        temporary_files_btn.click(
            confirmation_harness(delete_temp_files),
            inputs=tab_config.dummy_checkbox.instance,
            outputs=temporary_files_msg,
            js=confirm_box_js(
                "Удалить все временные файлы? Любые загруженные через UI файлы"
                " нужно будет загрузить снова перед повторной обработкой.",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Все временные файлы удалены!",
            ),
            outputs=temporary_files_msg,
            show_progress="hidden",
        )
