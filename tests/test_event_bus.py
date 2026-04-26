"""Tests for coding_agent.core.event_bus.EventBus."""
import asyncio
import pytest


async def test_emit_calls_handler():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    called = []
    bus.on("test", lambda x: called.append(x))
    bus.emit("test", 42)
    await asyncio.sleep(0)
    assert called == [42]


async def test_unsubscribe():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    called = []
    unsub = bus.on("test", lambda: called.append(1))
    unsub()
    bus.emit("test")
    assert called == []


async def test_async_handler():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    results = []

    async def handler(x):
        results.append(x)

    bus.on("ev", handler)
    bus.emit("ev", 99)
    await asyncio.sleep(0.05)
    assert results == [99]


def test_clear():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    called = []
    bus.on("x", lambda: called.append(1))
    bus.clear()
    bus.emit("x")
    assert called == []


def test_multiple_handlers_same_event():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    results = []
    bus.on("ev", lambda v: results.append(("a", v)))
    bus.on("ev", lambda v: results.append(("b", v)))
    bus.emit("ev", "hello")
    assert ("a", "hello") in results
    assert ("b", "hello") in results


def test_emit_no_handlers():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    # Should not raise
    bus.emit("nonexistent_event", 1, 2, 3)


def test_unsubscribe_twice_no_crash():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    called = []
    unsub = bus.on("ev", lambda: called.append(1))
    unsub()
    unsub()  # Should not raise
    bus.emit("ev")
    assert called == []


def test_multiple_events_independent():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    a_calls = []
    b_calls = []
    bus.on("a", lambda: a_calls.append(1))
    bus.on("b", lambda: b_calls.append(2))
    bus.emit("a")
    assert a_calls == [1]
    assert b_calls == []
    bus.emit("b")
    assert b_calls == [2]


async def test_handler_exception_does_not_crash_bus():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    ok_calls = []

    def bad_handler():
        raise RuntimeError("oops")

    bus.on("ev", bad_handler)
    bus.on("ev", lambda: ok_calls.append(1))
    bus.emit("ev")  # Should not raise
    await asyncio.sleep(0)
    assert ok_calls == [1]


def test_kwargs_passed_to_handler():
    from coding_agent.core.event_bus import EventBus
    bus = EventBus()
    received = {}
    bus.on("kw", lambda **kw: received.update(kw))
    bus.emit("kw", foo="bar", baz=42)
    assert received == {"foo": "bar", "baz": 42}
