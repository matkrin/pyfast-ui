from enum import Enum


class Channels(Enum):
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


channels = Channels("udi")
frame_range = 447


if channels.is_up_and_down():
    frame_id = image_range[0] * 2
else:
    frame_id = image_range[0]
for image_id in range(image_range[0], image_range[-1] + 1):
    for channel_id in self.channel_list:
        yield image_id, channel_id, frame_id
        frame_id += 1
