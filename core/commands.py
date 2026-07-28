"""
Command registry — all / commands are registered here.
Each command has: name, help text, aliases, handler function.
"""
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Command:
    name: str
    help: str
    handler: Callable[[str], None]  # takes the argument string after /command
    aliases: list[str] = field(default_factory=list)
    priority: int = 100
    is_visible: bool = True

# Global registry
_commands: dict[str, Command] = {}

def register(name_or_cmd, help=None, handler=None, aliases=None, priority=100):
    """Register a command. Accepts a Command object or individual args."""
    if isinstance(name_or_cmd, Command):
        cmd = name_or_cmd
    else:
        cmd = Command(name=name_or_cmd, help=help, handler=handler, aliases=aliases or [], priority=priority)
    _commands[cmd.name] = cmd
    for alias in cmd.aliases:
        _commands[alias] = cmd

def get(name: str) -> Command | None:
    return _commands.get(name)

def get_all() -> list[Command]:
    seen = set()
    result = []
    for cmd in _commands.values():
        if cmd.name not in seen:
            seen.add(cmd.name)
            result.append(cmd)
    result.sort(key=lambda c: c.priority)
    return result

def match(prefix: str) -> list[Command]:
    """Return commands matching a prefix string, sorted by priority."""
    seen = set()
    result = []
    p = prefix.lower()
    for cmd in _commands.values():
        if cmd.name not in seen and cmd.name.startswith(p):
            seen.add(cmd.name)
            result.append(cmd)
    result.sort(key=lambda c: c.priority)
    return result
