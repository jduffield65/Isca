import shelve
from joblib import dump, load
import inspect
import os
from typing import Optional, Union, Dict, Any


def save_workspace(
    filename: str,
    variables: Optional[Union[Dict[str, Any], list[str]]] = None,
    compress: Union[bool, str, tuple] = ("lz4", 3)
) -> None:
    """Save selected or all variables from the current workspace to a joblib file.

    If no variables are specified, saves all local variables from the caller's scope.

    Args:
        filename (str): Path to the `.joblib` file to save.
        variables (Optional[Union[Dict[str, Any], list[str]]]):
            Either:
            - A dictionary of variable names to values to save, or
            - A list of variable names (as strings) to extract from the caller's workspace.
            If None, all variables from the caller's local scope are saved.
        compress (Union[bool, str, tuple]):
            Compression method for joblib.
            Examples:
                - False (no compression)
                - 'lz4' or 'gzip'
                - ('lz4', 3) for method + compression level

    Example:
        >>> x, y = 1, [1, 2, 3]
        >>> save_workspace("test.joblib", ["x", "y"])
        >>> save_workspace("all_vars.joblib")  # saves everything in current workspace
    """
    # Get caller's frame
    frame = inspect.currentframe().f_back

    # If no variables provided, grab everything from local scope
    if variables is None:
        variables_to_save = frame.f_locals.copy()
    elif isinstance(variables, list):
        # Extract listed variable names from caller's local scope
        variables_to_save = {name: frame.f_locals[name] for name in variables if name in frame.f_locals}
    elif isinstance(variables, dict):
        variables_to_save = variables
    else:
        raise TypeError("`variables` must be a dict, list of variable names, or None")

    dump(variables_to_save, filename, compress=compress)

def load_workspace(path: str, target_globals: Optional[Dict[str, Any]] = None) -> None:
    """Load variables from a joblib file into the specified global namespace.

    Args:
        path (str): Path to the `.joblib` file containing the saved workspace.
        target_globals (Optional[Dict[str, Any]]):
            Dictionary (typically `globals()`) into which variables are loaded.
            Defaults to the caller's global namespace.

    Example:
        >>> load_workspace("workspace.joblib")
    """
    # Expand '~' and normalize path
    load_path = os.path.expanduser(path)

    # Get the caller's globals if none provided
    if target_globals is None:
        frame = inspect.currentframe().f_back
        target_globals = frame.f_globals

    # Load dictionary of saved variables
    data = load(load_path)

    if not isinstance(data, dict):
        raise ValueError("Joblib file does not contain a dictionary of variables.")

    # Inject into globals
    for name, value in data.items():
        target_globals[name] = value



# def save_workspace(path: str = "~/workspace_shelf", variables: Optional[dict] = None) -> None:
#     """Save all picklable variables to a shelve file.
#
#     Args:
#         path (str, optional): Path (without extension) to store the workspace.
#             Can include '~' for the home directory. Defaults to "~/workspace_shelf".
#         variables (dict, optional): Dictionary of variables to save.
#             If None, uses the caller's local variables.
#     """
#     import inspect
#
#     save_path = os.path.expanduser(path)
#     # Get caller's locals if not provided explicitly
#     if variables is None:
#         frame = inspect.currentframe().f_back
#         variables = frame.f_locals
#
#     with shelve.open(save_path) as shelf:
#         for k, v in variables.items():
#             try:
#                 shelf[k] = v
#             except Exception:
#                 print(f"Skipping {k} (not picklable)")


# def load_workspace(path: str = "~/workspace_shelf", target_globals: Optional[dict] = None) -> None:
#     """Load variables from a shelve file into the specified global namespace.
#
#     Args:
#         path (str, optional): Path (without extension) of the saved workspace.
#         target_globals (dict, optional): The globals() dictionary to populate.
#             Defaults to the caller's globals().
#     """
#     import inspect
#
#     load_path = os.path.expanduser(path)
#
#     if target_globals is None:
#         # Get the globals() of the caller
#         frame = inspect.currentframe().f_back
#         target_globals = frame.f_globals
#
#     with shelve.open(load_path) as shelf:
#         for k in shelf:
#             target_globals[k] = shelf[k]