import pytest
from energy_net.foundation.model import State, Action



def test_state_initialization_empty():
    s = State()
    assert len(s) == 0
    assert s.get_all_attributes() == {}


def test_state_initialization_with_values():
    s = State({"hp": 100, "mana": 50})
    assert len(s) == 2
    assert s.get_attribute("hp") == 100
    assert s.get_attribute("mana") == 50


def test_state_set_and_get_attribute():
    s = State()
    s.set_attribute("speed", 10)
    assert s.get_attribute("speed") == 10
    assert "speed" in s


def test_state_remove_attribute():
    s = State({"x": 1})
    s.remove_attribute("x")
    assert s.get_attribute("x") is None
    assert len(s) == 0


def test_state_clear_attributes():
    s = State({"a": 1, "b": 2})
    s.clear_attributes()
    assert len(s) == 0


def test_state_contains_and_len():
    s = State()
    s.set_attribute("alive", True)
    assert "alive" in s
    assert len(s) == 1



def test_action_initialization_empty():
    a = Action()
    assert len(a) == 0
    assert a.get_all_actions() == {}


def test_action_initialization_with_values():
    a = Action({"move": "left", "jump": True})
    assert len(a) == 2
    assert a.get_action("move") == "left"
    assert a.get_action("jump") is True


def test_action_set_and_get_action():
    a = Action()
    a.set_action("shoot", False)
    assert a.get_action("shoot") is False
    assert "shoot" in a


def test_action_remove_action():
    a = Action({"attack": 1})
    a.remove_action("attack")
    assert a.get_action("attack") is None
    assert len(a) == 0


def test_action_clear_actions():
    a = Action({"a": 1, "b": 2})
    a.clear_actions()
    assert len(a) == 0


def test_action_contains_and_len():
    a = Action()
    a.set_action("defend", True)
    assert "defend" in a
    assert len(a) == 1
