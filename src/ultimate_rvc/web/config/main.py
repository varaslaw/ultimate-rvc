"""
Module defining models for representing configuration settings for
UI tabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from functools import cached_property

from pydantic import BaseModel

from ultimate_rvc.typing_extra import SegmentSize, SeparationModel
from ultimate_rvc.web.config.component import (
    AnyComponentConfig,
    AudioConfig,
    CheckboxConfig,
    ComponentConfig,
    DropdownConfig,
    RadioConfig,
    SliderConfig,
)
from ultimate_rvc.web.config.tab import (
    SongGenerationConfig,
    SpeechGenerationConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    import gradio as gr


class SongIntermediateAudioConfig(BaseModel):
    """
    Configuration settings for intermediate audio components in the
    one-click song generation tab.

    Attributes
    ----------
    song : AudioConfig
        Configuration settings for the input song audio component.
    vocals : AudioConfig
        Configuration settings for the vocals audio component.
    instrumentals : AudioConfig
        Configuration settings for the instrumentals audio component.
    main_vocals : AudioConfig
        Configuration settings for the main vocals audio component.
    backup_vocals : AudioConfig
        Configuration settings for the backup vocals audio component.
    main_vocals_dereverbed : AudioConfig
        Configuration settings for the main vocals de-reverbed audio
        component.
    main_vocals_reverb : AudioConfig
        Configuration settings for the main vocals reverb audio
        component.
    converted_vocals : AudioConfig
        Configuration settings for the converted vocals audio
        component.
    postprocessed_vocals : AudioConfig
        Configuration settings for the postprocessed vocals audio
        component.
    instrumentals_shifted : AudioConfig
        Configuration settings for the shifted instrumentals audio
        component.
    backup_vocals_shifted : AudioConfig
        Configuration settings for the shifted backup vocals audio
        component.
    all : list[gr.Audio]
        List of instances of all intermediate audio components.

    """

    song: AudioConfig = AudioConfig.intermediate(label="Трек")
    vocals: AudioConfig = AudioConfig.intermediate(label="Вокал")
    instrumentals: AudioConfig = AudioConfig.intermediate(
        label="Инструментал",
    )
    main_vocals: AudioConfig = AudioConfig.intermediate(
        label="Основной вокал",
    )
    backup_vocals: AudioConfig = AudioConfig.intermediate(
        label="Бэк-вокал",
    )
    main_vocals_dereverbed: AudioConfig = AudioConfig.intermediate(
        label="Основной вокал без реверба",
    )
    main_vocals_reverb: AudioConfig = AudioConfig.intermediate(
        label="Основной вокал с ревербом",
    )
    converted_vocals: AudioConfig = AudioConfig.intermediate(
        label="Сконвертированный вокал",
    )
    postprocessed_vocals: AudioConfig = AudioConfig.intermediate(
        label="Постобработанный вокал",
    )
    instrumentals_shifted: AudioConfig = AudioConfig.intermediate(
        label="Инструментал со сдвигом тона",
    )
    backup_vocals_shifted: AudioConfig = AudioConfig.intermediate(
        label="Бэки со сдвигом тона",
    )

    @property
    def all(self) -> list[gr.Audio]:
        """
        Retrieve instances of all intermediate audio components
        in the one-click song generation tab.

        Returns
        -------
        list[gr.Audio]
            List of instances of all intermediate audio components in
            the one-click song generation tab.

        """
        # NOTE we are using self.__annotations__ to get the fields in
        # the order they are defined in the class
        return [getattr(self, field).instance for field in self.__annotations__]


class OneClickSongGenerationConfig(SongGenerationConfig):
    """
    Configuration settings for the one-click song generation tab.

    Attributes
    ----------
    n_octaves : SliderConfig
        Configuration settings for an octave pitch shift slider
        component.
    n_semitones : SliderConfig
        Configuration settings for a semitone pitch shift slider
        component.
    show_intermediate_audio : CheckboxConfig
        Configuration settings for a show intermediate audio checkbox
        component.
    intermediate_audio : SongIntermediateAudioConfig
        Configuration settings for intermediate audio components.

    See Also
    --------
    SongGenerationConfig
        Parent model defining common component configuration settings
        for song generation tabs.

    """

    n_octaves: SliderConfig = SliderConfig.octave_shift(
        label="Сдвиг высоты вокала",
        info=(
            "На сколько октав сдвигать высоту сконвертированного вокала. 1 — мужской"
            " → женский, -1 — женский → мужской."
        ),
    )

    n_semitones: SliderConfig = SliderConfig.semitone_shift(
        label="Общий сдвиг высоты",
        info=(
            "На сколько полутонов сдвигать высоту конвертированного вокала,"
            " инструментала и бэков."
        ),
    )
    show_intermediate_audio: CheckboxConfig = CheckboxConfig(
        label="Показывать промежуточные аудио",
        info="Отображать промежуточные дорожки, создаваемые в процессе генерации кавера.",
        value=False,
        exclude_value=True,
    )
    intermediate_audio: SongIntermediateAudioConfig = SongIntermediateAudioConfig()


class SongInputAudioConfig(BaseModel):
    """
    Configuration settings for input audio components in the multi-step
    song generation tab.

    Attributes
    ----------
    audio : AudioConfig
        Configuration settings for the input audio component.
    vocals : AudioConfig
        Configuration settings for the vocals audio component.
    converted_vocals : AudioConfig
        Configuration settings for the converted vocals audio
        component.
    instrumentals : AudioConfig
        Configuration settings for the instrumentals audio
        component.
    backup_vocals : AudioConfig
        Configuration settings for the backup vocals audio
        component.
    main_vocals : AudioConfig
        Configuration settings for the main vocals audio
        component.
    shifted_instrumentals : AudioConfig
        Configuration settings for the shifted instrumentals audio
        component.
    shifted_backup_vocals : AudioConfig
        Configuration settings for the shifted backup vocals audio
        component.
    all : list[AudioConfig]
        List of configuration settings for all input audio
        components in the multi-step song generation tab.

    """

    audio: AudioConfig = AudioConfig.input(label="Аудио")
    vocals: AudioConfig = AudioConfig.input(label="Вокал")
    converted_vocals: AudioConfig = AudioConfig.input(label="Вокал")
    instrumentals: AudioConfig = AudioConfig.input(label="Инструментал")
    backup_vocals: AudioConfig = AudioConfig.input(label="Бэк-вокал")
    main_vocals: AudioConfig = AudioConfig.input(label="Основной вокал")
    shifted_instrumentals: AudioConfig = AudioConfig.input(label="Инструментал")
    shifted_backup_vocals: AudioConfig = AudioConfig.input(label="Бэк-вокал")

    @property
    def all(self) -> list[AudioConfig]:
        """
        Retrieve configuration settings for all input audio components
        in the multi-step song generation tab.

        Returns
        -------
        list[AudioConfig]
            List of configuration settings for all input audio
            components in the multi-step song generation tab.

        """
        return [getattr(self, field) for field in self.__annotations__]


class SongDirsConfig(BaseModel):
    """
    Configuration settings for song directory components in the
    multi-step song generation tab.

    Attributes
    ----------
    separate_audio : DropdownConfig
        Configuration settings for the song directory component
        for separating audio.
    convert_vocals : DropdownConfig
        Configuration settings for the song directory component
        for converting vocals.
    postprocess_vocals : DropdownConfig
        Configuration settings for the song directory component
        for postprocessing vocals.
    pitch_shift_background : DropdownConfig
        Configuration settings for the song directory component
        for pitch-shifting background audio.
    mix : DropdownConfig
        Configuration settings for the song directory component
        for mixing audio.
    all : list[gr.Dropdown]
        List of instances of all song directory components in the
        multi-step song generation tab.

    """

    separate_audio: DropdownConfig = DropdownConfig.song_dir()
    convert_vocals: DropdownConfig = DropdownConfig.song_dir()
    postprocess_vocals: DropdownConfig = DropdownConfig.song_dir()
    pitch_shift_background: DropdownConfig = DropdownConfig.song_dir()
    mix: DropdownConfig = DropdownConfig.song_dir()

    @property
    def all(self) -> list[gr.Dropdown]:
        """
        Retrieve instances of all song directory components in the
        multi-step song generation tab.

        Returns
        -------
        list[gr.Dropdown]
            List of instances of all song directory components in
            the multi-step song generation tab.

        """
        return [getattr(self, field).instance for field in self.__annotations__]


class MultiStepSongGenerationConfig(SongGenerationConfig):
    """
    Configuration settings for multi-step song generation tab.

    Attributes
    ----------
    separation_model : DropdownConfig
        Configuration settings for a separation model dropdown
        component.
    segment_size : RadioConfig
        Configuration settings for a segment size radio component.
    n_octaves : SliderConfig
        Configuration settings for an octave pitch shift slider
        component.
    n_semitones : SliderConfig
        Configuration settings for a semitone pitch shift slider
        component.
    n_semitones_instrumentals : SliderConfig
        Configuration settings for an instrumentals pitch shift slider
        component.
    n_semitones_backup_vocals : SliderConfig
        Configuration settings for a backup vocals pitch shift slider
        component.
    input_audio : SongInputAudioConfig
        Configuration settings for input audio components.
    song_dirs : SongDirsConfig
        Configuration settings for song directory components.

    See Also
    --------
    SongGenerationConfig
        Parent model defining common component configuration settings
        for song generation tabs.

    """

    separation_model: DropdownConfig = DropdownConfig(
        label="Модель разделения",
        info="Модель, используемая для разделения аудио на стемы.",
        value=SeparationModel.UVR_MDX_NET_VOC_FT,
        choices=list(SeparationModel),
    )
    segment_size: RadioConfig = RadioConfig(
        label="Размер сегмента",
        info=(
            "Длина фрагментов, на которые делится аудио. Больше — расходует больше"
            " ресурсов, но может дать лучшее качество."
        ),
        value=SegmentSize.SEG_512,
        choices=list(SegmentSize),
    )
    n_octaves: SliderConfig = SliderConfig.octave_shift(
        label="Сдвиг тона (октавы)",
        info=(
            "На сколько октав смещать высоту конвертированного голоса. 1 —"
            " мужской → женский, -1 — наоборот."
        ),
    )
    n_semitones: SliderConfig = SliderConfig.semitone_shift(
        label="Сдвиг тона (полутона)",
        info=(
            "На сколько полутонов сдвигать высоту конвертированного вокала."
            " Сильное изменение может слегка ухудшить качество."
        ),
    )
    n_semitones_instrumentals: SliderConfig = SliderConfig.semitone_shift(
        label="Сдвиг тона инструментала",
        info="На сколько полутонов смещать высоту инструментала.",
    )
    n_semitones_backup_vocals: SliderConfig = SliderConfig.semitone_shift(
        label="Сдвиг тона бэков",
        info="На сколько полутонов смещать высоту бэк-вокала.",
    )
    input_audio: SongInputAudioConfig = SongInputAudioConfig()
    song_dirs: SongDirsConfig = SongDirsConfig()


class SpeechIntermediateAudioConfig(BaseModel):
    """
    Configuration settings for intermediate audio components in the
    one-click speech generation tab.

    Attributes
    ----------
    speech : AudioConfig
        Configuration settings for the input speech audio component.
    converted_speech : AudioConfig
        Configuration settings for the converted speech audio component.
    all : list[gr.Audio]
        List of instances of all intermediate audio components in the
        speech generation tab.

    """

    speech: AudioConfig = AudioConfig.intermediate(label="Речь")
    converted_speech: AudioConfig = AudioConfig.intermediate(label="Сконвертированная речь")

    @property
    def all(self) -> list[gr.Audio]:
        """
        Retrieve instances of all intermediate audio components in the
        speech generation tab.

        Returns
        -------
        list[gr.Audio]
            List of instances of all intermediate audio components in
            the speech generation tab.

        """
        return [getattr(self, field).instance for field in self.__annotations__]


class OneClickSpeechGenerationConfig(SpeechGenerationConfig):
    """
    Configuration settings for one-click speech generation tab.

    Attributes
    ----------
    intermediate_audio : SpeechIntermediateAudioConfig
        Configuration settings for intermediate audio components.
    show_intermediate_audio : CheckboxConfig
        Configuration settings for a show intermediate audio checkbox
        component.

    See Also
    --------
    SpeechGenerationConfig
        Parent model defining common component configuration settings
        for speech generation tabs.

    """

    intermediate_audio: SpeechIntermediateAudioConfig = SpeechIntermediateAudioConfig()

    show_intermediate_audio: CheckboxConfig = CheckboxConfig(
        label="Показывать промежуточные аудио",
        info="Отображать промежуточные дорожки, созданные при генерации речи.",
        value=False,
        exclude_value=True,
    )


class SpeechInputAudioConfig(BaseModel):
    """
    Configuration settings for input audio components in the multi-step
    speech generation tab.

    Attributes
    ----------
    speech : AudioConfig
        Configuration settings for the input speech audio component.
    converted_speech : AudioConfig
        Configuration settings for the converted speech audio component.

    all : list[AudioConfig]
        List of configuration settings for all input audio components in
        the multi-step speech generation tab.

    """

    speech: AudioConfig = AudioConfig.input("Речь")
    converted_speech: AudioConfig = AudioConfig.input("Сконвертированная речь")

    @property
    def all(self) -> list[AudioConfig]:
        """
        Retrieve configuration settings for all input audio components
        in the multi-step speech generation tab.

        Returns
        -------
        list[AudioConfig]
            List of configuration settings for all input audio
            components in the multi-step speech generation tab.

        """
        return [getattr(self, field) for field in self.__annotations__]


class MultiStepSpeechGenerationConfig(SpeechGenerationConfig):
    """
    Configuration settings for the multi-step speech generation tab.

    Attributes
    ----------
    input_audio : SpeechInputAudioConfig
        Configuration settings for input audio components.

    See Also
    --------
    SpeechGenerationConfig
        Parent model defining common component configuration settings
        for speech generation tabs.

    """

    input_audio: SpeechInputAudioConfig = SpeechInputAudioConfig()


class MultiStepTrainingConfig(TrainingConfig):
    """Configuration settings for multi-step training tab."""


class ModelManagementConfig(BaseModel):
    """

    Configuration settings for model management tab.

    Attributes
    ----------
    voices : DropdownConfig
        Configuration settings for delete voice models dropdown
        component.
    embedders : DropdownConfig
        Configuration settings for delete embedder models dropdown
        component.
    pretraineds : DropdownConfig
        Configuration settings for delete pretrained models dropdown
        component.
    traineds : DropdownConfig
        Configuration settings for delete training models dropdown
        component.
    dummy_checkbox : CheckboxConfig
        Configuration settings for a dummy checkbox component.

    """

    voices: DropdownConfig = DropdownConfig.multi_delete(
        label="Голосовые модели",
        info="Выберите одну или несколько голосовых моделей для удаления.",
    )
    embedders: DropdownConfig = DropdownConfig.multi_delete(
        label="Пользовательские эмбеддеры",
        info="Выберите пользовательские модели эмбеддеров для удаления.",
    )
    pretraineds: DropdownConfig = DropdownConfig.multi_delete(
        label="Пользовательские предобученные модели",
        info="Выберите предобученные модели для удаления.",
    )
    traineds: DropdownConfig = DropdownConfig.multi_delete(
        label="Обученные модели",
        info="Выберите обученные модели для удаления.",
    )

    dummy_checkbox: CheckboxConfig = CheckboxConfig(
        value=False,
        visible=False,
        exclude_value=True,
    )


class AudioManagementConfig(BaseModel):
    """
    Configuration settings for audio management tab.

    Attributes
    ----------
    intermediate : DropdownConfig
        Configuration settings for delete intermediate audio files
        dropdown component
    speech : DropdownConfig
        Configuration settings for delete speech audio files dropdown
        component.
    output : DropdownConfig
        Configuration settings for delete output audio files dropdown
        component.
    dataset : DropdownConfig
        Configuration settings for delete dataset audio files dropdown
        component.
    dummy_checkbox : CheckboxConfig
        Configuration settings for a dummy checkbox component.

    """

    intermediate: DropdownConfig = DropdownConfig.multi_delete(
        label="Папки песен",
        info=(
            "Выберите одну или несколько папок с промежуточными файлами, которые"
            " нужно удалить."
        ),
    )
    speech: DropdownConfig = DropdownConfig.multi_delete(
        label="Аудио речи",
        info="Выберите один или несколько файлов речи для удаления.",
    )
    output: DropdownConfig = DropdownConfig.multi_delete(
        label="Готовые аудио",
        info="Выберите выходные аудиофайлы для удаления.",
    )
    dataset: DropdownConfig = DropdownConfig.multi_delete(
        label="Аудио датасетов",
        info="Выберите наборы данных с аудиофайлами, которые нужно удалить.",
    )

    dummy_checkbox: CheckboxConfig = CheckboxConfig(
        value=False,
        visible=False,
        exclude_value=True,
    )


class SettingsManagementConfig(BaseModel):
    """
    Configuration settings for settings management tab.

    Attributes
    ----------
    dummy_checkbox : CheckboxConfig
        Configuration settings for a dummy checkbox component.

    """

    load_config_name: DropdownConfig = DropdownConfig(
        label="Имя конфигурации",
        info="Конфигурация, из которой загрузить настройки интерфейса",
        value=None,
        render=False,
        exclude_value=True,
    )
    delete_config_names: DropdownConfig = DropdownConfig.multi_delete(
        label="Имена конфигураций",
        info="Выберите одну или несколько конфигураций для удаления",
    )
    dummy_checkbox: CheckboxConfig = CheckboxConfig(
        value=False,
        visible=False,
        exclude_value=True,
    )


class TotalSongGenerationConfig(BaseModel):
    """
    All configuration settings for song generation tabs.

    Attributes
    ----------
    one_click : OneClickSongGenerationConfig
        Configuration settings for the one-click song generation tab.
    multi_step : MultiStepSongGenerationConfig
        Configuration settings for the multi-step song generation tab.

    """

    one_click: OneClickSongGenerationConfig = OneClickSongGenerationConfig()
    multi_step: MultiStepSongGenerationConfig = MultiStepSongGenerationConfig()


class TotalSpeechGenerationConfig(BaseModel):
    """
    All configuration settings for speech generation tabs.

    Attributes
    ----------
    one_click : OneClickSpeechGenerationConfig
        Configuration settings for the one-click speech generation tab.
    multi_step : MultiStepSpeechGenerationConfig
        Configuration settings for the multi-step speech generation tab.

    """

    one_click: OneClickSpeechGenerationConfig = OneClickSpeechGenerationConfig()
    multi_step: MultiStepSpeechGenerationConfig = MultiStepSpeechGenerationConfig()


class TotalTrainingConfig(BaseModel):
    """
    All configuration settings for training tabs.

    Attributes
    ----------
    training : TrainingConfig
        Configuration settings for the multi-step training tab.

    """

    multi_step: MultiStepTrainingConfig = MultiStepTrainingConfig()


class TotalManagementConfig(BaseModel):
    """
    All configuration settings for management tabs.

    Attributes
    ----------
    model : ModelManagementConfig
        Configuration settings for the model management tab.
    audio : AudioManagementConfig
        Configuration settings for the audio management tab.
    settings : SettingsManagementConfig
        Configuration settings for the settings management tab.

    """

    model: ModelManagementConfig = ModelManagementConfig()
    audio: AudioManagementConfig = AudioManagementConfig()
    settings: SettingsManagementConfig = SettingsManagementConfig()


class TotalConfig(BaseModel):
    """
    All configuration settings for the Ultimate RVC app.

    Attributes
    ----------
    song : TotalSongGenerationConfig
        Configuration settings for song generation tabs.
    speech : TotalSpeechGenerationConfig
        Configuration settings for speech generation tabs.
    training : TotalTrainingConfig
        Configuration settings for training tabs.
    management : TotalManagementConfig
        Configuration settings for management tabs.

    """

    song: TotalSongGenerationConfig = TotalSongGenerationConfig()
    speech: TotalSpeechGenerationConfig = TotalSpeechGenerationConfig()
    training: TotalTrainingConfig = TotalTrainingConfig()
    management: TotalManagementConfig = TotalManagementConfig()

    @cached_property
    def all(self) -> list[AnyComponentConfig]:
        """
        Recursively collect those component configuration models nested
        within the current model instance, which have values that are
        not excluded.

        Returns
        -------
        list[AnyComponentConfig]
            A list of component configuration models found within the
            current model instance, which have values that are not
            excluded.

        """

        def _collect(model: BaseModel) -> list[AnyComponentConfig]:
            component_configs: list[Any] = []
            for _, value in model:
                if isinstance(value, ComponentConfig):
                    if not value.exclude_value:
                        component_configs.append(value)
                elif isinstance(value, BaseModel):
                    component_configs.extend(_collect(value))
            return component_configs

        return _collect(self)
