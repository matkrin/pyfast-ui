from __future__ import annotations

import itertools
from enum import Enum
from typing import Literal, TypeAlias


FrameChannelType: TypeAlias = Literal["uf", "ub", "ui", "df", "db", "di"]


class Channels(Enum):
    """Channels of a `FastMovie`. Indicate which subset of collected data
    points are used.

    UDI: Up and down frames, forward and backward scanning interlaced.
    UDF: Up and down frames, just forward scanning.
    UDB: Up and down frames, just backward scanning.
    UF: Up frames, forwards scanning.
    UB: Up frames, backwards scanning.
    DF: Down frames, forwards scanning.
    DB: Down frames, backwards scanning.
    UI: Up frames, forward and backward scanning interlaced.
    DI: Down frames, forward and backward scanning interlaced.
    """

    UDI = "udi"
    UDF = "udf"
    UDB = "udb"
    UF = "uf"
    UB = "ub"
    DF = "df"
    DB = "db"
    UI = "ui"
    DI = "di"

    def is_interlaced(self) -> bool:
        return "i" in self.value

    def is_forward(self) -> bool:
        return "f" in self.value

    def is_backward(self) -> bool:
        return "b" in self.value

    def is_up_and_down(self) -> bool:
        return "u" in self.value and "d" in self.value

    def is_up_not_down(self) -> bool:
        return "u" in self.value and "d" not in self.value

    def is_down_not_up(self) -> bool:
        return "d" in self.value and "u" not in self.value

    def frame_channel_iterator(
        self,
    ) -> itertools.cycle[FrameChannelType]:
        """Gives an cyclic iterator over single frames used, according to
        selected channel of a movie.
        """
        cycle_list: list[FrameChannelType] = []
        match self:
            case Channels.UDI:
                cycle_list = ["ui", "di"]
            case Channels.UDF:
                cycle_list = ["uf", "df"]
            case Channels.UDB:
                cycle_list = ["ub", "db"]
            case _:
                cycle_list = [self.value]

        return itertools.cycle(cycle_list)
