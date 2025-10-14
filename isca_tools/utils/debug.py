import shelve

import shelve
import os
from typing import Optional


def save_workspace(path: str = "~/workspace_shelf", variables: Optional[dict] = None) -> None:
    """Save all picklable variables to a shelve file.

    Args:
        path (str, optional): Path (without extension) to store the workspace.
            Can include '~' for the home directory. Defaults to "~/workspace_shelf".
        variables (dict, optional): Dictionary of variables to save.
            If None, uses the caller's local variables.
    """
    import inspect

    save_path = os.path.expanduser(path)
    # Get caller's locals if not provided explicitly
    if variables is None:
        frame = inspect.currentframe().f_back
        variables = frame.f_locals

    with shelve.open(save_path) as shelf:
        for k, v in variables.items():
            try:
                shelf[k] = v
            except Exception:
                print(f"Skipping {k} (not picklable)")


def load_workspace(path: str = "~/workspace_shelf", target_globals: Optional[dict] = None) -> None:
    """Load variables from a shelve file into the specified global namespace.

    Args:
        path (str, optional): Path (without extension) of the saved workspace.
        target_globals (dict, optional): The globals() dictionary to populate.
            Defaults to the caller's globals().
    """
    import inspect

    load_path = os.path.expanduser(path)

    if target_globals is None:
        # Get the globals() of the caller
        frame = inspect.currentframe().f_back
        target_globals = frame.f_globals

    with shelve.open(load_path) as shelf:
        for k in shelf:
            target_globals[k] = shelf[k]