from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_rate_hz", int(max(1, int(self.sample_rate_hz))))
        object.__setattr__(self, "channels", int(max(1, int(self.channels))))
        object.__setattr__(self, "sample_width_bytes", int(max(1, int(self.sample_width_bytes))))

    @property
    def bytes_per_second(self) -> int:
        return int(self.sample_rate_hz * self.channels * self.sample_width_bytes)

    def ms_to_byte_offset(self, ms: int) -> int:
        raw = int(round((max(0.0, float(ms)) / 1000.0) * float(self.bytes_per_second)))
        align = int(max(1, self.sample_width_bytes))
        if (raw % align) != 0:
            raw -= raw % align
        return int(max(0, raw))

    def bytes_to_ms(self, byte_count: int) -> int:
        aligned = int(max(0, int(byte_count)))
        align = int(max(1, self.sample_width_bytes))
        if (aligned % align) != 0:
            aligned -= aligned % align
        return int((aligned * 1000) // max(1, self.bytes_per_second))
