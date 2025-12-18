"""Module defining common component configurations for UI tabs."""

from __future__ import annotations

from pydantic import BaseModel

from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioNormalizationMode,
    AudioSplitMethod,
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PrecisionType,
    PretrainedType,
    SampleRate,
    TrainingSampleRate,
    Vocoder,
)
from ultimate_rvc.web.config.component import (
    CheckboxConfig,
    DropdownConfig,
    NumberConfig,
    SliderConfig,
    TextboxConfig,
)
from ultimate_rvc.web.typing_extra import DatasetType, SongSourceType, SpeechSourceType


class BaseTabConfig(BaseModel):
    """
    Base model defining common component configuration settings for
    UI tabs.

    Attributes
    ----------
    embedder_model : DropdownConfig
        Configuration settings for an embedder model dropdown component.
    custom_embedder_model : DropdownConfig
        Configuration settings for a custom embedder model dropdown
        component.

    """

    embedder_model: DropdownConfig = DropdownConfig(
        label="Модель эмбеддера",
        info="Модель, которая используется для построения голосовых эмбеддингов.",
        value=EmbedderModel.CONTENTVEC,
        choices=list(EmbedderModel),
        exclude_value=True,
    )
    custom_embedder_model: DropdownConfig = DropdownConfig(
        label="Пользовательская модель эмбеддера",
        info="Выберите пользовательскую модель эмбеддера из списка.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )


class GenerationConfig(BaseTabConfig):
    """
    Common component configuration settings for generation tabs.

    voice_model : DropdownConfig
        Configuration settings for a voice model dropdown component.
    f0_method : DropdownConfig
        Configuration settings for a pitch extraction algorithm
        dropdown component.
    index_rate : SliderConfig
        Configuration settings for an index rate slider component.
    rms_mix_rate : SliderConfig
        Configuration settings for a RMS mix rate slider component.
    protect_rate : SliderConfig
        Configuration settings for a protect rate slider component.
    split_voice : CheckboxConfig
        Configuration settings for a split voice checkbox component.
    autotune_voice: CheckboxConfig
        Configuration settings for an autotune voice checkbox component.
    autotune_strength: SliderConfig
        Configuration settings for an autotune strength slider
        component.
    proposed_pitch: CheckboxConfig
        Configuration settings for a proposed pitch checkbox component.
    proposed_pitch_threshold: SliderConfig
        Configuration settings for a proposed pitch threshold slider
        component.
    sid : NumberConfig
        Configuration settings for a speaker ID number component.
    output_sr : DropdownConfig
        Configuration settings for an output sample rate dropdown
        component.
    output_format : DropdownConfig
        Configuration settings for an output format dropdown
        component.
    output_name : TextboxConfig
        Configuration settings for an output name textbox component.

    See Also
    --------
    BaseTabConfig
        Parent model defining common component configuration settings
        for UI tabs.

    """

    voice_model: DropdownConfig = DropdownConfig(
        label="Голосовая модель",
        info="Выберите модель, которая будет использоваться для конверсии голоса.",
        value=None,
        render=False,
        exclude_value=True,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="Алгоритм извлечения высоты",
        info=(
            "RMVPE рекомендуем по умолчанию: он быстро и точнее всего извлекает"
            " высоту для большинства случаев."
        ),
        value=F0Method.RMVPE,
        choices=list(F0Method),
        multiselect=False,
    )
    index_rate: SliderConfig = SliderConfig(
        label="Сила индекса",
        info=(
            "Чем выше значение, тем сильнее конверсия стремится к акценту модели."
            " Уменьшение может снизить артефакты, приходящие из модели"
            " голоса.<br><br><br>"
        ),
        value=0.3,
        minimum=0.0,
        maximum=1.0,
    )
    rms_mix_rate: SliderConfig = SliderConfig(
        label="Смешивание RMS",
        info=(
            "Насколько сохранять громкость исходного голоса (0) или приводить её"
            " к фиксированной громкости (1). Значение 1 рекомендовано в большинстве"
            " случаев.<br><br>"
        ),
        value=1.0,
        minimum=0.0,
        maximum=1.0,
    )
    protect_rate: SliderConfig = SliderConfig(
        label="Степень защиты",
        info=(
            "Определяет, насколько активно защищать согласные и дыхание от артефактов."
            " Чем выше значение, тем больше защита, но может ухудшиться эффект"
            " индексации.<br><br>"
        ),
        value=0.33,
        minimum=0.0,
        maximum=0.5,
    )

    split_voice: CheckboxConfig = CheckboxConfig(
        label="Делить входной голос",
        info=(
            "Разделять ли входную дорожку на мелкие сегменты перед конверсией."
            " Это может улучшить качество для длинных треков."
        ),
        value=False,
    )
    autotune_voice: CheckboxConfig = CheckboxConfig(
        label="Автотюн для конвертированного голоса",
        info="Применять ли автотюн к сконвертированному голосу.",
        value=False,
        exclude_value=True,
    )
    autotune_strength: SliderConfig = SliderConfig(
        label="Интенсивность автотюна",
        info=(
            "Высокие значения сильнее привязывают ноты к хроматической сетке и"
            " могут добавить артефакты."
        ),
        value=1.0,
        minimum=0.0,
        maximum=1.0,
        visible=False,
    )
    proposed_pitch: CheckboxConfig = CheckboxConfig(
        label="Предложенная высота",
        info=(
            "Настраивать ли высоту конвертированного голоса под диапазон выбранной"
            " модели."
        ),
        value=False,
        exclude_value=True,
    )
    proposed_pitch_threshold: SliderConfig = SliderConfig(
        label="Порог предложенной высоты",
        info=(
            "Для мужских моделей обычно 155.0, для женских — примерно 255.0."
        ),
        value=155.0,
        minimum=50.0,
        maximum=1200.0,
        visible=False,
    )
    sid: NumberConfig = NumberConfig(
        label="ID спикера",
        info="Идентификатор спикера для многоголосовых моделей.",
        value=0,
        precision=0,
    )
    output_sr: DropdownConfig = DropdownConfig(
        label="Частота дискретизации вывода",
        info="Частота дискретизации итоговой смешанной дорожки.",
        value=SampleRate.HZ_44K,
        choices=list(SampleRate),
    )
    output_format: DropdownConfig = DropdownConfig(
        label="Формат вывода",
        info="Аудиоформат итоговой дорожки.",
        value=AudioExt.MP3,
        choices=list(AudioExt),
    )
    output_name: TextboxConfig = TextboxConfig(
        label="Имя файла",
        info="Если имя не указано, подходящее название будет создано автоматически.",
        value=None,
        placeholder="Выходной файл Ultimate RVC AISingers RUS",
        exclude_value=True,
    )


class SongGenerationConfig(GenerationConfig):
    """
    Common component configuration settings for song generation tabs.

    Attributes
    ----------
    source_type : DropdownConfig
        Configuration settings for a source type dropdown component.
    source : TextboxConfig
        Configuration settings for an input source textbox component.
    cached_song : DropdownConfig
        Configuration settings for a cached song dropdown component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider component.
    clean_voice : CheckboxConfig
        Configuration settings for a clean voice checkbox component.
    room_size : SliderConfig
        Configuration settings for a room size slider component.
    wet_level : SliderConfig
        Configuration settings for a wetness level slider component.
    dry_level : SliderConfig
        Configuration settings for a dryness level slider component.
    damping : SliderConfig
        Configuration settings for a damping level slider component.
    main_gain : SliderConfig
        Configuration settings for a main gain slider component.
    inst_gain : SliderConfig
        Configuration settings for an instrumentals gain slider
        component.
    backup_gain : SliderConfig
        Configuration settings for a backup vocals gain slider
        component.

    See Also
    --------
    GenerationConfig
        Parent model defining common component configuration settings
        for song generation tabs.

    """

    source_type: DropdownConfig = DropdownConfig(
        label="Тип источника",
        info="Откуда брать трек для кавера.",
        value=SongSourceType.PATH,
        choices=list(SongSourceType),
        type="index",
        exclude_value=True,
    )
    source: TextboxConfig = TextboxConfig(
        label="Источник",
        info="Ссылка на трек в YouTube или полный путь к локальному аудиофайлу.",
        value=None,
        exclude_value=True,
    )
    cached_song: DropdownConfig = DropdownConfig(
        label="Источник",
        info="Выберите трек из списка уже загруженных.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Очистка конвертированного голоса",
        info="Применять ли шумоподавление к сконвертированному голосу.",
        value=False,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=False)
    room_size: SliderConfig = SliderConfig(
        label="Размер помещения",
        info=(
            "Размер пространства, которое симулирует реверберация. Увеличьте для"
            " более длинного хвоста."
        ),
        value=0.15,
        minimum=0.0,
        maximum=1.0,
    )
    wet_level: SliderConfig = SliderConfig(
        label="Уровень wet",
        info="Громкость обработанных голосов с реверберацией.",
        value=0.2,
        minimum=0.0,
        maximum=1.0,
    )
    dry_level: SliderConfig = SliderConfig(
        label="Уровень dry",
        info="Громкость обработанных голосов без реверберации.",
        value=0.8,
        minimum=0.0,
        maximum=1.0,
    )
    damping: SliderConfig = SliderConfig(
        label="Затухание",
        info="Поглощение высоких частот в реверберации.",
        value=0.7,
        minimum=0.0,
        maximum=1.0,
    )
    main_gain: SliderConfig = SliderConfig.gain(
        label="Громкость основного голоса",
        info="Усиление для основной вокальной партии.",
    )
    inst_gain: SliderConfig = SliderConfig.gain(
        label="Громкость инструментала",
        info="Усиление для инструментала.",
    )
    backup_gain: SliderConfig = SliderConfig.gain(
        label="Громкость бэков",
        info="Усиление для бэк-вокала.",
    )


class SpeechGenerationConfig(GenerationConfig):
    """
    Common component configuration settings for speech generation tabs.

    Attributes
    ----------
    source_type : DropdownConfig
        Configuration settings for a source type dropdown component.
    source : TextboxConfig
        Configuration settings for an input source textbox component.
    edge_tts_voice : DropdownConfig
        Configuration settings for an Edge TTS voice dropdown
        component.
    n_octaves : SliderConfig
        Configuration settings for an octave pitch shift slider
        component.
    n_semitones : SliderConfig
        Configuration settings for a semitone pitch shift slider
        component.
    tts_pitch_shift : SliderConfig
        Configuration settings for a TTS pitch shift slider
        component.
    tts_speed_change : SliderConfig
        Configuration settings for a TTS speed change slider
        component.
    tts_volume_change : SliderConfig
        Configuration settings for a TTS volume change slider
        component.
    clean_voice : CheckboxConfig
        Configuration settings for a clean voice checkbox
        component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider
        component.
    output_gain : GainSliderConfig
        Configuration settings for an output gain slider component.

    See Also
    --------
    GenerationConfig
        Parent model defining common component configuration settings
        for generation tabs.

    """

    source_type: DropdownConfig = DropdownConfig(
        label="Тип источника",
        info="Откуда брать текст или аудио для генерации речи.",
        value=SpeechSourceType.TEXT,
        choices=list(SpeechSourceType),
        type="index",
        exclude_value=True,
    )
    source: TextboxConfig = TextboxConfig(
        label="Источник",
        info="Текст, который нужно озвучить",
        value=None,
        exclude_value=True,
    )
    edge_tts_voice: DropdownConfig = DropdownConfig(
        label="Голос Edge TTS",
        info="Выберите голос для синтеза речи через Edge TTS.",
        value=None,
        render=False,
        exclude_value=True,
    )
    n_octaves: SliderConfig = SliderConfig.octave_shift(
        label="Сдвиг по октавам",
        info=(
            "Количество октав, на которое смещается высота конвертированной речи."
            " 1 — мужской → женский, -1 — наоборот."
        ),
    )
    n_semitones: SliderConfig = SliderConfig.semitone_shift(
        label="Сдвиг по полутонам",
        info="Количество полутонов для смещения высоты конвертированной речи.",
    )
    tts_pitch_shift: SliderConfig = SliderConfig(
        label="Сдвиг высоты Edge TTS",
        info=(
            "На сколько герц смещать высоту речи, созданной Edge TTS, ещё до"
            " конверсии."
        ),
        value=0,
        minimum=-100,
        maximum=100,
        step=1,
    )
    tts_speed_change: SliderConfig = SliderConfig(
        label="Скорость TTS",
        info="Изменение скорости речи Edge TTS в процентах.",
        value=0,
        minimum=-50,
        maximum=100,
        step=1,
    )
    tts_volume_change: SliderConfig = SliderConfig(
        label="Громкость TTS",
        info="Процентное изменение громкости речи, сгенерированной Edge TTS.",
        value=0,
        minimum=-100,
        maximum=100,
        step=1,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Очистка конвертированного голоса",
        info="Применять ли шумоподавление к сконвертированной речи.",
        value=True,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=True)
    output_gain: SliderConfig = SliderConfig.gain(
        label="Громкость вывода",
        info="Усиление, применяемое к итоговой речи.<br><br>",
    )


class TrainingConfig(BaseTabConfig):
    """
    Common component configuration settings for training tabs.

    Attributes
    ----------
    dataset_type : DropdownConfig
        Configuration settings for a dataset type dropdown component.
    dataset : DropdownConfig
        Configuration settings for a dataset dropdown component.
    dataset_name : TextboxConfig
        Configuration settings for a dataset name textbox component.
    preprocess_model : DropdownConfig
        Configuration settings for a model name dropdown component
        for audio preprocessing.
    sample_rate : DropdownConfig
        Configuration settings for a sample rate dropdown component.
    normalization_mode: DropdownConfig
        Configuration settings for a normalization mode dropdown
        component.
    filter_audio : CheckboxConfig
        Configuration settings for a filter audio checkbox component.
    clean_audio : CheckboxConfig
        Configuration settings for a clean audio checkbox component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider component.
    split_method : DropdownConfig
        Configuration settings for an audio splitting method dropdown
        component.
    chunk_len : SliderConfig
        Configuration settings for a chunk length slider component.
    overlap_len : SliderConfig
        Configuration settings for an overlap length slider component.
    preprocess_cores : SliderConfig
        Configuration settings for a CPU cores slider component for
        preprocessing.
    extract_model : DropdownConfig
        Configuration settings for a model name dropdown component for
        feature extraction.
    f0_method : DropdownConfig
        Configuration settings for an F0 method dropdown component.
    include_mutes : SliderConfig
        Configuration settings for an include mutes slider component.
    extract_cores : SliderConfig
        Configuration settings for a CPU cores slider component for
        feature extraction.
    extraction_acceleration : HardwareAccelerationConfig
        Configuration settings for a hardware acceleration component for
        feature extraction.
    extraction_gpus : DropdownConfig
        Configuration settings for a GPU dropdown compoennt for feature
        extraction.
    train_model : DropdownConfig
        Configuration settings for a model name dropdown component for
        training.
    num_epochs : SliderConfig
        Configuration settings for a number of epochs slider component.
    batch_size : SliderConfig
        Configuration settings for a batch size slider component.
    detect_overtraining : CheckboxConfig
        Configuration settings for a detect overtraining checkbox
        component.
    overtraining_threshold : SliderConfig
        Configuration settings for an overtraining threshold slider
        component.
    vocoder : DropdownConfig
        Configuration settings for a vocoder dropdown component.
    index_algorithm : DropdownConfig
        Configuration settings for an index algorithm dropdown
        component.
    pretrained_type : DropdownConfig
        Configuration settings for a pretrained model type dropdown
        component.
    custom_pretrained_model : DropdownConfig
        Configuration settings for a custom pretrained model dropdown
        component.
    save_interval : SliderConfig
        Configuration settings for a save-interval slider component.
    save_all_checkpoints : CheckboxConfig
        Configuration settings for a save-all-checkpoints checkbox
        component.
    save_all_weights : CheckboxConfig
        Configuration settings for a save-all-weights checkbox
        component.
    clear_saved_data : CheckboxConfig
        Configuration settings for a clear-saved-data checkbox
        component.
    upload_model : CheckboxConfig
        Configuration settings for an upload voice model checkbox
        component.
    upload_name : TextboxConfig
        Configuration settings for an upload name textbox component.
    training_acceleration : HardwareAccelerationConfig
        Configuration settings for a hardware acceleration component for
        training.
    training_gpus : DropdownConfig
        Configuration settings for a GPU dropdown component for
        training.
    precision: DropdownConfig
        Configuration settings for a precision type dropdown component.
    preload_dataset : CheckboxConfig
        Configuration settings for a preload dataset checkbox component.
    reduce_memory_usage : CheckboxConfig
        Configuration settings for a reduce-memory-usage checkbox
        component.

    See Also
    --------
    BaseTabConfig
        Parent model defining common component configuration settings
        for UI tabs.

    """

    dataset_type: DropdownConfig = DropdownConfig(
        label="Тип датасета",
        info="Выберите тип датасета, который хотите подготовить.",
        value=DatasetType.NEW_DATASET,
        choices=list(DatasetType),
        exclude_value=True,
    )
    dataset: DropdownConfig = DropdownConfig(
        label="Путь к датасету",
        info=(
            "Путь к существующему датасету. Можно выбрать ранее созданный набор"
            " или указать внешний путь."
        ),
        value=None,
        allow_custom_value=True,
        visible=False,
        render=False,
        exclude_value=True,
    )
    dataset_name: TextboxConfig = TextboxConfig(
        label="Имя датасета",
        info=(
            "Название нового датасета. Если он уже существует, выбранные аудио"
            " будут к нему добавлены."
        ),
        value="My dataset",
        exclude_value=True,
    )
    preprocess_model: DropdownConfig = DropdownConfig(
        label="Имя модели",
        info=(
            "Название модели, под которую будет готовиться датасет. Можно выбрать"
            " существующую или ввести новую."
        ),
        value="My model",
        allow_custom_value=True,
        render=False,
        exclude_value=True,
    )
    sample_rate: DropdownConfig = DropdownConfig(
        label="Частота дискретизации",
        info="Целевая частота дискретизации для аудио в датасете.",
        value=TrainingSampleRate.HZ_40K,
        choices=list(TrainingSampleRate),
    )
    normalization_mode: DropdownConfig = DropdownConfig(
        label="Режим нормализации",
        info="Метод нормализации, применяемый к аудио в датасете.",
        value=AudioNormalizationMode.POST,
        choices=list(AudioNormalizationMode),
    )
    filter_audio: CheckboxConfig = CheckboxConfig(
        label="Фильтрация аудио",
        info=(
            "Удалять ли низкочастотные шумы с помощью ВЧ-фильтра Баттерворта"
            " для файлов в датасете.<br><br>"
        ),
        value=True,
    )
    clean_audio: CheckboxConfig = CheckboxConfig(
        label="Очистка аудио",
        info=(
            "Применять ли шумоподавление к аудио в датасете.<br><br><br>"
        ),
        value=False,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=False)
    split_method: DropdownConfig = DropdownConfig(
        label="Метод нарезки аудио",
        info=(
            "Как делить аудио в датасете. `Skip` — пропустить, если файлы уже"
            " нарезаны. `Simple` — если лишние паузы удалены. `Automatic` —"
            " автоматически искать тишину и резать вокруг неё."
        ),
        value=AudioSplitMethod.AUTOMATIC,
        choices=list(AudioSplitMethod),
        exclude_value=True,
    )
    chunk_len: SliderConfig = SliderConfig(
        label="Длина чанка",
        info="Длительность нарезанных фрагментов.",
        value=3.0,
        minimum=0.5,
        maximum=5.0,
        step=0.1,
        visible=False,
    )
    overlap_len: SliderConfig = SliderConfig(
        label="Длина перекрытия",
        info="Размер перекрытия между соседними фрагментами.",
        value=0.3,
        minimum=0.0,
        maximum=0.4,
        step=0.1,
        visible=False,
    )
    preprocess_cores: SliderConfig = SliderConfig.cpu_cores()

    extract_model: DropdownConfig = DropdownConfig(
        label="Имя модели",
        info=(
            "Название модели с подготовленным датасетом, из которого нужно"
            " извлечь фичи. После новой подготовки датасета его модель выбирается"
            " автоматически."
        ),
        value=None,
        render=False,
        exclude_value=True,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="Метод F0",
        info="Метод извлечения высоты тона (pitch).",
        value=F0Method.RMVPE,
        choices=list(F0Method),
        exclude_value=True,
    )

    include_mutes: SliderConfig = SliderConfig(
        label="Количество тишины",
        info=(
            "Сколько файлов с тишиной добавить в список для обучения. Это помогает"
            " модели справляться с паузами. Если в датасете уже есть тихие"
            " сегменты, поставьте 0."
        ),
        value=2,
        minimum=0,
        maximum=10,
        step=1,
    )
    extraction_cores: SliderConfig = SliderConfig.cpu_cores()
    extraction_acceleration: DropdownConfig = DropdownConfig.hardware_acceleration()
    extraction_gpus: DropdownConfig = DropdownConfig.gpu()

    train_model: DropdownConfig = DropdownConfig(
        label="Имя модели",
        info=(
            "Название модели для обучения. После извлечения фич для новой модели"
            " её имя выбирается автоматически."
        ),
        value=None,
        render=False,
        exclude_value=True,
    )
    num_epochs: SliderConfig = SliderConfig(
        label="Количество эпох",
        info=(
            "Сколько эпох обучать голосовую модель. Больше эпох — лучше качество,"
            " но выше риск переобучения."
        ),
        value=500,
        minimum=1,
        maximum=1000,
        step=1,
    )
    batch_size: SliderConfig = SliderConfig(
        label="Batch size",
        info=(
            "Количество сэмплов в каждом батче. Подберите значение под доступную"
            " видеопамять."
        ),
        value=8,
        minimum=1,
        maximum=64,
        step=1,
    )
    detect_overtraining: CheckboxConfig = CheckboxConfig(
        label="Отслеживать переобучение",
        info=(
            "Включить контроль переобучения, чтобы модель не запоминала датасет"
            " дословно и не теряла обобщение."
        ),
        value=False,
        exclude_value=True,
    )
    overtraining_threshold: SliderConfig = SliderConfig(
        label="Порог переобучения",
        info=(
            "Максимум эпох без улучшения качества, после которых тренировка"
            " прекращается."
        ),
        value=50,
        minimum=1,
        maximum=100,
        visible=False,
        step=1,
    )
    vocoder: DropdownConfig = DropdownConfig(
        label="Вокодер",
        info=(
            "Вокодер для синтеза аудио во время обучения. HiFi-GAN даёт базовое"
            " качество, RefineGAN — максимально высокое."
        ),
        value=Vocoder.HIFI_GAN,
        choices=list(Vocoder),
    )
    index_algorithm: DropdownConfig = DropdownConfig(
        label="Алгоритм индексации",
        info=(
            "Метод построения индекс-файла для обученной голосовой модели."
            " `KMeans` особенно полезен на больших датасетах."
        ),
        value=IndexAlgorithm.AUTO,
        choices=list(IndexAlgorithm),
    )
    pretrained_type: DropdownConfig = DropdownConfig(
        label="Тип предобученной модели",
        info=(
            "Какой предобученный вес использовать для дообучения. `None` — с нуля,"
            " `Default` — стандартный под архитектуру, `Custom` — ваш собственный"
            " вес."
        ),
        value=PretrainedType.DEFAULT,
        choices=list(PretrainedType),
        exclude_value=True,
    )
    custom_pretrained_model: DropdownConfig = DropdownConfig(
        label="Пользовательская предобученная модель",
        info="Выберите пользовательский предобученный вес для дообучения.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )
    save_interval: SliderConfig = SliderConfig(
        label="Интервал сохранения",
        info=(
            "Через сколько эпох сохранять веса и контрольные точки. Лучшие веса"
            " сохраняются всегда, независимо от этого параметра."
        ),
        value=10,
        minimum=1,
        maximum=100,
        step=1,
    )
    save_all_checkpoints: CheckboxConfig = CheckboxConfig(
        label="Сохранять все чекпоинты",
        info=(
            "Сохранять отдельный чекпоинт на каждом шаге. Если выкл., хранится"
            " только последний."
        ),
        value=False,
    )
    save_all_weights: CheckboxConfig = CheckboxConfig(
        label="Сохранять все веса",
        info=(
            "Сохранять отдельные веса модели на каждом интервале. Если выкл.,"
            " остаются только лучшие веса."
        ),
        value=False,
    )
    clear_saved_data: CheckboxConfig = CheckboxConfig(
        label="Очистить сохранённые данные",
        info=(
            "Удалять ли старые данные обучения, связанные с моделью, перед стартом."
            " Включайте, если начинаете с нуля или перезапускаете обучение."
        ),
        value=False,
    )
    upload_model: CheckboxConfig = CheckboxConfig(
        label="Загрузить голосовую модель",
        info=(
            "Автоматически выгрузить обученную модель, чтобы использовать её в"
            " Ultimate RVC AISingers RUS."
        ),
        value=False,
        exclude_value=True,
    )
    upload_name: TextboxConfig = TextboxConfig(
        label="Имя при загрузке",
        info="Название, под которым будет загружена голосовая модель.",
        value=None,
        visible=False,
        exclude_value=True,
    )
    training_acceleration: DropdownConfig = DropdownConfig.hardware_acceleration()
    training_gpus: DropdownConfig = DropdownConfig.gpu()
    precision: DropdownConfig = DropdownConfig(
        label="Точность вычислений",
        info=(
            "Тип точности при обучении. FP16 и BF16 снижают потребление VRAM и"
            " ускоряют обучение на поддерживаемом железе."
        ),
        value=PrecisionType.FP32,
        choices=list(PrecisionType),
    )
    preload_dataset: CheckboxConfig = CheckboxConfig(
        label="Предзагружать датасет",
        info=(
            "Загружать ли все данные обучения в видеопамять. Ускоряет обучение, но"
            " требует много VRAM.<br><br>"
        ),
        value=False,
    )
    reduce_memory_usage: CheckboxConfig = CheckboxConfig(
        label="Экономить память",
        info=(
            "Включить активационный чекпоинтинг для экономии VRAM ценой скорости."
            " Полезно для карт с ограниченной памятью (<6 ГБ) или при большом"
            " batch size."
        ),
        value=False,
    )
