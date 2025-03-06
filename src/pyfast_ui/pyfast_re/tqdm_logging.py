from collections.abc import Iterable, Iterator
import logging
from typing import Any, TypeVar, final, override

from tqdm import tqdm


T = TypeVar("T")


@final
class TqdmLogger(Iterable[T]):
    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        desc: str | None = None,
        total: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> None:
        self.logger: logging.Logger = logging.getLogger()
        self.show_progress: bool = self.logger.level <= logging.INFO
        self.kwargs: dict[str, Any] = kwargs  # pyright: ignore[reportExplicitAny]
        self.iterable = iterable
        self.desc = desc
        self.total = total
        self.tqdm_instance: tqdm[T] | None = None

    @override
    def __iter__(self) -> Iterator[T]:
        if self.show_progress:
            self.tqdm_instance = tqdm(
                self.iterable,
                desc=self.desc,
                total=self.total,
                **self.kwargs,  # pyright: ignore[reportAny]
            )
            return iter(self.tqdm_instance)

        else:
            if self.iterable is None:
                return iter(())  # Return empty iterator

            return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        if self.show_progress and self.tqdm_instance:
            _ = self.tqdm_instance.update(n)

    def close(self) -> None:
        if self.show_progress and self.tqdm_instance:
            self.tqdm_instance.close()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if self.show_progress and self.tqdm_instance:
            self.tqdm_instance.set_description(desc, refresh)

    def set_postfix(self, **kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        if self.show_progress and self.tqdm_instance:
            self.tqdm_instance.set_postfix(**kwargs)  # pyright: ignore[reportAny, reportUnknownMemberType]
