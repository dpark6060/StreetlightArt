"""Maps pygame-gui slider widgets to SceneConfig field paths."""

import typing as t
from dataclasses import dataclass, field


@dataclass
class SliderDescriptor:
    """Describes the mapping between a slider widget and a config field.

    Attributes:
        label: Human-readable label shown next to the slider.
        config_path: Tuple of keys/indices to traverse the nested config.
            String items are attribute names; integer items are list indices.
            This is also the path used when reading the initial value.
        min_val: Slider minimum value.
        max_val: Slider maximum value.
        is_int: When True, the value is cast to int before writing to config.
        extra_config_paths: Additional paths to write to when applying this
            slider's value. Used when a single slider controls the same field
            across multiple pool entries (e.g. all fixture profiles for a road
            type). Reads always come from config_path; writes go to config_path
            and all entries in extra_config_paths.
    """

    label: str
    config_path: tuple
    min_val: float
    max_val: float
    is_int: bool = False
    extra_config_paths: t.List[tuple] = field(default_factory=list)


class SliderRegistry:
    """Holds all widget-to-config-path mappings for a single control panel section.

    Each registry lives for the lifetime of a displayed section. On section
    change the old registry is discarded and a new one is created after killing
    all tracked widgets.

    Attributes:
        sliders: Maps UIHorizontalSlider → SliderDescriptor.
        value_entries: Maps UIHorizontalSlider → UITextEntryLine (editable value display).
        entry_to_slider: Reverse lookup — maps UITextEntryLine → UIHorizontalSlider.
        bool_buttons: Maps UIButton → config_path tuple.
        bool_states: Maps UIButton → current bool value.
        text_entries: Maps UITextEntryLine → config_path tuple (for non-slider string fields).
        extra_widgets: Any widgets (e.g. row labels) not tracked in the above dicts.
    """

    def __init__(self) -> None:
        """Initialise empty widget mapping dicts."""
        self.sliders: t.Dict[t.Any, SliderDescriptor] = {}
        self.value_entries: t.Dict[t.Any, t.Any] = {}
        self.entry_to_slider: t.Dict[t.Any, t.Any] = {}
        self.bool_buttons: t.Dict[t.Any, tuple] = {}
        self.bool_states: t.Dict[t.Any, bool] = {}
        self.text_entries: t.Dict[t.Any, tuple] = {}
        self.extra_widgets: t.List[t.Any] = []

    def register_slider(
        self,
        slider: t.Any,
        descriptor: SliderDescriptor,
        value_entry: t.Optional[t.Any] = None,
    ) -> None:
        """Register a slider widget and its optional value text entry.

        Args:
            slider: A UIHorizontalSlider instance.
            descriptor (SliderDescriptor): Metadata describing the slider's config binding.
            value_entry: Optional UITextEntryLine showing (and accepting) the current value.
        """
        self.sliders[slider] = descriptor
        if value_entry is not None:
            self.value_entries[slider] = value_entry
            self.entry_to_slider[value_entry] = slider

    def register_bool(self, button: t.Any, config_path: tuple, initial_value: bool = False) -> None:
        """Register a toggle button that controls a boolean config field.

        Args:
            button: A UIButton instance acting as an ON/OFF toggle.
            config_path (tuple): Path to the boolean field within SceneConfig.
            initial_value (bool): The current boolean value from config.
        """
        self.bool_buttons[button] = config_path
        self.bool_states[button] = initial_value

    def register_text(self, entry: t.Any, config_path: tuple) -> None:
        """Register a text entry that controls a string config field.

        Args:
            entry: A UITextEntryLine instance.
            config_path (tuple): Path to the string field within SceneConfig.
        """
        self.text_entries[entry] = config_path

    def all_widgets(self) -> t.List[t.Any]:
        """Return every tracked widget so callers can kill them all at once.

        Returns:
            list: All sliders, value entries, bool buttons, text entries, and extra widgets.
        """
        widgets: t.List[t.Any] = []
        widgets.extend(self.sliders.keys())
        widgets.extend(self.value_entries.values())
        widgets.extend(self.bool_buttons.keys())
        widgets.extend(self.text_entries.keys())
        widgets.extend(self.extra_widgets)
        return widgets
