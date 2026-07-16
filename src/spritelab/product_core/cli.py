"""Instance-scoped CLI extension registry used by product plugins."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

CliInstaller = Callable[[argparse._SubParsersAction], None]


class CliRegistrationError(ValueError):
    """A CLI extension attempted an ambiguous or invalid registration."""


@dataclass(frozen=True)
class CliCommandRegistration:
    name: str
    owner: str
    installer: CliInstaller


class ProductCliRegistry:
    """A disposable registry assembled by the integration layer for one parser."""

    def __init__(self, *, parents: Sequence[argparse.ArgumentParser] = ()) -> None:
        self._commands: dict[str, CliCommandRegistration] = {}
        self._parents = tuple(parents)

    def register(self, name: str, installer: CliInstaller, *, owner: str, replace: bool = False) -> None:
        normalized = name.strip().lower()
        if not normalized or " " in normalized:
            raise CliRegistrationError(f"Invalid CLI command name: {name!r}")
        existing = self._commands.get(normalized)
        if existing is not None and not replace:
            raise CliRegistrationError(
                f"CLI command {normalized!r} is already registered by {existing.owner!r}; "
                "an intentional replacement must pass replace=True."
            )
        self._commands[normalized] = CliCommandRegistration(normalized, owner, installer)

    def command(
        self,
        name: str,
        *,
        owner: str,
        handler: Callable[..., object],
        help: str,
        configure: Callable[[argparse.ArgumentParser], None] | None = None,
        replace: bool = False,
    ) -> None:
        """Register a conventional product command with shared output flags."""

        def install(target: argparse._SubParsersAction) -> None:
            parser = target.add_parser(name, parents=list(self._parents), help=help)
            if configure:
                configure(parser)
            parser.set_defaults(handler=handler)

        self.register(name, install, owner=owner, replace=replace)

    def contains(self, name: str) -> bool:
        return name in self._commands

    def install(self, subparsers: argparse._SubParsersAction) -> None:
        for command in self._commands.values():
            before = set(subparsers.choices)
            command.installer(subparsers)
            added = set(subparsers.choices) - before
            if command.name not in added:
                raise CliRegistrationError(
                    f"Installer owned by {command.owner!r} did not add its declared command {command.name!r}."
                )

    def __iter__(self) -> Iterator[CliCommandRegistration]:
        return iter(self._commands.values())
