# utils/system_control.py

def set_system_volume(percent: int):
    # TODO: implement with pycaw (Windows) or other OS-specific API
    print(f"[SYSTEM] Set volume to {percent}% (stub)")

def set_screen_brightness(percent: int):
    # TODO: implement with screen-brightness-control or OS API
    print(f"[SYSTEM] Set brightness to {percent}% (stub)")

def toggle_play_pause():
    print("[SYSTEM] Toggle play/pause (stub)")

def next_track():
    print("[SYSTEM] Next track (stub)")

def previous_track():
    print("[SYSTEM] Previous track (stub)")

def open_application(name: str):
    print(f"[SYSTEM] Open application: {name} (stub)")
