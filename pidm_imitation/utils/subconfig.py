# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import os
from typing import List, Type, TypeVar

from schema import Schema, SchemaError

T = TypeVar("T")


class SubConfig(ABC):
    """
    Abstract class that is used to read a block of parameters from the config file.
    The child classes of ``SubConfig`` receive a dictionary referring to a specific
    block in the config file. Therefore, these classes are very specific to a certain
    set of parameters: one SubConfig class for the observation block, another class for
    data on the reference trajectory, and so on. These classes parse that dictionary
    and check if it follows the schema required (obtained through the abstract method
    ``_get_schema()``), and also set some internal attributes when needed.

    The child classes of ``SubConfig`` are not meant to be instantiated directly in the
    scripts, but rather by the ``ConfigFile`` classes, which are the classes that read
    a config file, break it into separate blocks of parameters, and then call the
    correct ``SubConfig`` child class for each of these blocks.

    Every child class should also define the KEY attribute whenever the subclass is
    meant to directly read a block of parameters from the config file. The KEY attribute
    defines the name of the key used to access that block of parameters in the config file
    (after that config file is converted to a dictionary).
    """

    POSITIVE_NUMBER_ERR = "The value assigned to this key must be a positive number."
    NON_NEGATIVE_ERR = "The value assigned to this key must be a non-negative number."
    PARAM_RANGE_01_ERR = "This parameter must be a float between 0 and 1."
    RADIUS_ERR = "the radius must be a positive float"

    def __init__(self, config_path: str, config: dict) -> None:
        """
        :param config_path: path to the config file this subconfig is referring to.
        :param config: the dictionary representing a block of parameters extracted from
        the config file.
        """
        self._config_path = os.path.realpath(config_path)
        self._config = config
        self._validate_schema()

    @property
    def config_dict(self) -> dict:
        return self._config

    @property
    def config_path(self) -> str:
        return self._config_path

    @property
    def base_path(self) -> str:
        return os.path.dirname(self._config_path)

    def _run_schema_validation(self, config: dict, schema_format: Schema) -> None:
        try:
            schema_format.validate(config)
        except SchemaError as se:
            raise se

    @abstractmethod
    def _get_schema(self) -> Schema:
        pass

    def _validate_schema(self) -> None:
        if self._config is None:
            raise ValueError(f"ERROR: The mandatory parser {self.__class__.__name__} was not provided.")

        schema = self._get_schema()
        self._run_schema_validation(self._config, schema)

    def _get_simple_config_obj(self, sub_config_cls: Type[T]) -> T | None:
        """
        This method receives any SubConfig class reference and returns an object of that class. First, it checks if the
        SubConfig subclass reference provided has the KEY attribute (the KEY attribute is used by child classes of
        ``SubConfig`` to specify what is the key used by a set of parameters in the config file). It uses that key to
        access the full config dictionary (self._config). If the config dictionary doesn't have that key, return None.
        It then builds the class object and returns it. Notice that this method only work for "simple" config parser
        subclasses, which are the subclasses that only require the ``config`` and the ``config_path``
        parameters. This method won't work for subclasses that require more parameters than those two.

        :param sub_config_cls: reference to a class. This is the class that we want to build, and so the type of this
            class should be the same as the return type;
        :return: an object of the class specified by ``sub_config_cls``
        """
        assert hasattr(sub_config_cls, "KEY"), "ERROR: the class provided requires a KEY attribute."
        key = getattr(sub_config_cls, "KEY")
        config_dict = self._config.get(key, None)
        if config_dict is None:
            return None
        return sub_config_cls(config_path=self._config_path, config=config_dict)  # type: ignore

    def _get_simple_config_list(self, sub_config_cls: Type[T]) -> List[T] | None:
        """
        This method receives a list of SubConfig class references and returns a list of objects of that class. First, it
        checks if the SubConfig subclass reference provided has the KEY attribute (the KEY attribute is used by child
        classes of ``SubConfig`` to specify what is the key used by a set of parameters in the config file). It uses
        that key to access the full config dictionary (self._config). If the config dictionary doesn't have that key,
        return None. It then walks the given list in the config_dict key and constructs each class object and returns
        the resulting list. Notice that this method only work for "simple" config parser subclasses, which are the
        subclasses that only require the ``config`` and the ``config_path`` parameters. This method won't
        work for subclasses that require more parameters than those two.

        :param sub_config_cls: reference to a class. This is the class that we want to build, and so the type of this
            class should be the same as the return type;
        :return: an object of the class specified by ``sub_config_cls``
        """
        assert hasattr(sub_config_cls, "KEY"), "ERROR: the class provided requires a KEY attribute."
        key = getattr(sub_config_cls, "KEY")
        config_dict = self._config.get(key, None)
        if config_dict is None:
            return None
        if not isinstance(config_dict, list):
            raise ValueError(f"ERROR: the config dictionary for {key} must be a list.")
        result = []
        for sub_config in config_dict:
            item = sub_config_cls(config_path=self._config_path, config=sub_config)  # type: ignore
            result.append(item)
        return result

    def get_subconfig_att(self, sub_config_cls: Type[T]) -> T | None:
        """
        Retrieves the first attribute of ``self`` that matches the type passed
        as a parameter. This is used to retrieve an internal sub-config object
        when the exact type of ``self`` is known, that is, when we only know
        that the config objet is of type ``ConfigFile``, but the exact sub-class
        is unknown. This method is called then to get a sub-config object, when
        it's present. Otherwise, return ``None``. The return type is the same as
        the type passed in the parameter ``sub_config_cls``.

        :param sub_config_cls: reference to a class. We want to check if ``self``
            has any attributes with this type. In case it does, return the first
            attribute with that type. Otherwise, return ``None``;
        :return: the first attribute whose type matches ``sub_config_cls``, or
            ``None`` if no attribute matches that type.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) and isinstance(attr, sub_config_cls):
                return attr
        return None
