from pathlib import Path


def test_gui_launcher_exists():
    launcher = Path("/home/macko/Desktop/TEST_llama-optimus/scripts/launch_llama_optimus_gui.sh")
    assert launcher.is_file()


def test_desktop_entry_exists():
    desktop_entry = Path("/home/macko/Desktop/TEST_llama-optimus/TEST_llama-optimus.desktop")
    assert desktop_entry.is_file()
