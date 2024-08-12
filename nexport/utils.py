# COPYRIGHT NOTICE

# “Neural Network Export Package (nexport) v0.4.6” Copyright (c) 2023,
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.

# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights. As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do so.


# Module imports
import nexport
import sys
import os
# File imports
from nexport.pytorch import exporting as npte
from nexport.pytorch import importing as npti
from nexport.tensorflow import exporting as ntfe
from nexport.tensorflow import importing as ntfi

# External function visibility
__all__ = ['detect_framework', 'export', 'nimport']


# Module functions

def detect_framework(imported: object = sys.modules.keys()) -> str:
    """
    Detect deep learning framework
    in use from imported modules.
    """
    frameworks = {
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "tensorflow-macos": "tensorflow",
        "tensorflow-metal": "tensorflow"
    }

    detected_module = [module for module in frameworks.keys() if module in imported]
    
    if len(detected_module) == 1:
        return frameworks[detected_module[0]]
    elif len(detected_module) > 1:
        return "multiple"
    else:
        return "none"


def export(model: object, filetype: str, input_size: int, output_size: int, filename: str = "model", indent: int = 4, verbose: int = 1, include_metadata: bool = False, model_name: str = "My Model", model_author: str = None, activation_function: str = None, using_skip_connections: bool = None, intercept: float = 0.0, slope: float = 1.0) -> None:
    match nexport.__framework__:
        case "pytorch":
            match filetype:
                case "txt":
                    npte.export_to_file(model=model, filename=filename)
                case "json":
                    npte.export_to_json(model=model, filename=filename, indent=indent, verbose=verbose, include_metadata=include_metadata, model_name=model_name, model_author=model_author, activation_function=activation_function.lower(), using_skip_connections=using_skip_connections)
                case "json_exp":
                    # npte.export_to_json_experimental(model=model, filename=filename, indent=indent,
                    # verbose=verbose, include_metadata=include_metadata, model_name=model_name,
                    # model_author=model_author, activation_function=activation_function.lower(),
                    # using_skip_connections=using_skip_connections)
                    npte.export_to_json_experimental(model=model, filename=filename, indent=indent, verbose=verbose,
                                                     include_metadata=include_metadata, model_name=model_name,
                                                     model_author=model_author, activation_function=activation_function.lower(),
                                                     using_skip_connections=using_skip_connections,
                                                     input_size=input_size, output_size=output_size,
                                                     intercept=intercept, slope=slope)
                case "csv" | "xml":
                        raise NotImplementedError(f"This feature (exporting {nexport.__framework__} in {filetype}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")
        case "tensorflow" | "tensorflow-macos" | "tensorflow-metal":
            match filetype:
                case "txt" | "json" | "csv" | "xml":
                    raise NotImplementedError(f"This feature (exporting {nexport.__framework__} in {filetype}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")


def nimport(filepath: str, verbose: int = 1) -> object:
    extension = os.path.splitext(filepath)[-1]
    match nexport.__framework__:
        case "pytorch":
            match extension:
                case ".txt" | "":
                    return npti.import_from_file(filepath=filepath, verbose=verbose)
                case ".json":
                    raise NotImplementedError(f"This feature (importing {extension} to {nexport.__framework__}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")
        case "tensorflow" | "tensorflow-macos" | "tensorflow-metal":
            match extension:
                case ".txt" | "" | ".json" | ".csv" | ".xml":
                    raise NotImplementedError(f"This feature (importing {extension} to {nexport.__framework__}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")
