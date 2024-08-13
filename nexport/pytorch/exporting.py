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
import json
# import os # disbaling os.getlogin() as it causes issues on supercomputers
import datetime as dt
import time as t

# File imports
from nexport.colors import Color as c

# External function visibillity
__all__ = ['export_to_file', 'export_to_json', 'export_to_json_experimental']


# Module functions

def export_to_file(model: object, filename: str = "model") -> None:
    """
    Function which exports all weight
    and bias arrays to a file.
    """
    filename = nexport.append_extension(filename=filename, extension='txt')
    print(f"Creating file: {c.YELLOW}{filename}{c.DEFAULT}")
    f = open(filename, "w")
    for x, y in enumerate(model.parameters()):  # access parameter array
        if (x % 2 != 1):  # if array is not odd (even index = weights)
            print(f"Extracting {c.GREEN}weights{c.DEFAULT} from layer {c.RED}{(int(x / 2)) + 1}{c.DEFAULT}")
            for i, z in enumerate(y):  # access weight array x in layer y
                f.write("[") if i == 0 else f.write(" ")
                weights = []
                for w in z:  # access weights for neuron w in from neuron z in layer y
                    weights.append(float(w))
                f.write(str(weights))
                f.write("]") if i == len(y) - 1 else f.write("")
                f.write("\n")
            f.write("\n")
        else:  # if array is odd (odd index = biases)
            print(f"Extracting {c.MAGENTA}biases{c.DEFAULT} from layer  {c.RED}{(int(x / 2)) + 1}{c.DEFAULT}")
            biases = []
            for b in y:
                biases.append(float(b))
            f.write(str(biases))
            f.write("\n\n\n")
    print(f"Saving file: {c.YELLOW}{filename}{c.DEFAULT}")
    f.close()
    print(f"{c.CYAN}Done!{c.DEFAULT}")


def create_paramater_arrays(model: object, verbose: int = None) -> tuple:
    """
    Function which splits a model's state_dict into weight
    and bias arrays and returns them for indexing elsewhere.
    """
    model_dictionary = model.state_dict()
    weights = []
    biases = []

    # Loop which creates parameter arrays from model's state_dict
    for x, item in enumerate(model_dictionary):
        if x % 2 == 0:  # if even (weight)
            weights.append(model_dictionary[item])
        else:  # if odd (bias)
            biases.append(model_dictionary[item])

    if verbose >= 2:  # if verbose set to at least 2
        print(f"{c.RED}Successfully extracted {c.LIGHTRED}parameters.{c.DEFAULT}")

    return weights, biases  # return weights & biases as tuple


def create_layer_object(weights: list, biases: list, verbose: int = None) -> list:
    """
    Function which constructs a single layer from
    parameter arrays and returns it as a list of neurons.
    """
    neuron_list = []
    temp_weights = []
    temp_bias = 0
    temp_dict = {}

    # Loop which creates a layer as a list from parameter arrays
    for i in range(len(weights)):
        for j in weights[i]:
            temp_weights.append(j.item())
        temp_bias = biases[i].item()
        temp_dict["weights"] = temp_weights.copy()
        temp_dict["bias"] = temp_bias
        neuron_list.append(temp_dict.copy())
        temp_weights.clear()
        temp_dict.clear()

    if verbose >= 3:  # if verbose set to at least 3
        print(f"{c.LIGHTYELLOW}    Layer created.{c.DEFAULT}")

    return neuron_list  # return constructed layer


def create_model_metadata(model_name: str, model_author: str = None, activation_function: str = None,
                          using_skip_connections: bool = None) -> dict:
    model_metadata = {
        "modelName": model_name,
        "modelAuthor": model_author,
        "compilationDate": str(dt.datetime.now()),
        "activationFunction": activation_function,
        "usingSkipConnections": using_skip_connections
    }

    return model_metadata  # return model metadata object


def create_acceptable_inference_engine_release_metadata(acceptable_inference_engine_release: str) -> dict:
    return {"acceptable_engine_tag": acceptable_inference_engine_release}


def create_data_normalization_metadata(data_length: int, intercept: float = 0.0, slope: float = 1.0,
                                       layer_label: str = None) -> dict:
    model_metadata = {
        "layer": layer_label,
        "intercept": [intercept for _ in range(data_length)],
        "slope": [slope for _ in range(data_length)],
    }

    return model_metadata  # return model metadata object


def create_model_object(model: object, verbose: int = None, include_metadata: bool = None, model_name: str = None,
                        model_author: str = None, activation_function: str = None,
                        using_skip_connections: bool = None) -> object:
    """
    Function which creates a model object from a
    collection of layers instantiated with layer
    objects (neuron lists).
    """
    hidden_layers = []
    output_layer = []
    model_object = {}
    weights, biases = create_paramater_arrays(model=model, verbose=verbose)

    if include_metadata:  # insert model metadata into model object
        model_object["metadata"] = create_model_metadata(model_name=model_name, model_author=model_author,
                                                         activation_function=activation_function,
                                                         using_skip_connections=using_skip_connections)

    if verbose >= 3:  # if verbose set to at least 3
        print(f"{c.YELLOW}Creating layers...{c.DEFAULT}")

    # Loop which creates a network object from a series of single-layer lists
    for x, layer in enumerate(weights):
        if x != len(weights) - 1:
            hidden_layers.append(create_layer_object(weights=weights[x], biases=biases[x], verbose=verbose))
        else:
            output_layer = create_layer_object(weights=weights[x], biases=biases[x], verbose=verbose)

    model_object["hidden_layers"] = hidden_layers
    model_object["output_layer"] = output_layer

    if verbose >= 2:  # if verbose set to at least 2
        print(f"{c.GREEN}Successfully created {c.LIGHTGREEN}model object.{c.DEFAULT}")

    return model_object  # return constructed network


def export_to_json(model: object, filename: str = None, indent: int = None, verbose: int = None,
                   include_metadata: bool = None, model_name: str = None, model_author: str = None,
                   activation_function: str = None, using_skip_connections: bool = None) -> None:
    """
    Function which exports a passed model
    object to a JSON file.
    """
    t1 = t.time()
    model_object = {}
    if include_metadata:
        model_object = create_model_object(model=model, verbose=verbose, include_metadata=include_metadata,
                                           model_name=model_name, model_author=model_author,
                                           activation_function=activation_function,
                                           using_skip_connections=using_skip_connections)
    else:
        model_object = create_model_object(model=model, verbose=verbose)
    json_object = json.dumps(obj=model_object, indent=indent)

    with open(nexport.append_extension(filename=filename, extension="json"), "w") as outfile:
        outfile.write(json_object)

    t2 = t.time()
    time = t2 - t1

    if verbose >= 1:  # if verbose set to at least 1
        print(
            f"{c.CYAN}Exported model to {c.LIGHTCYAN}'{nexport.append_extension(filename=filename, extension='json')}'{c.CYAN}!{c.DEFAULT}")
    if verbose >= 2:  # if verbose set to at least 2
        print(f"{c.MAGENTA}    Time taken: {c.LIGHTMAGENTA}{round(time, 2)}{c.MAGENTA}s{c.DEFAULT}")


def export_to_json_experimental(model: object, acceptable_inference_engine_release: str,
                                filename: str = None, indent: int = None, verbose: int = None,
                                include_metadata: bool = None, model_name: str = None, model_author: str = None,
                                activation_function: str = None, using_skip_connections: bool = None,
                                input_size: int = None, output_size: int = None, intercept: float = 0.0,
                                slope: float = 1.0) -> None:
    """
    Function which exports a passed
    model object to a JSON file, but
    keeps array elements on one line.
    """
    t1 = t.time()

    model_inference_engine_release_metadata = create_acceptable_inference_engine_release_metadata(
        acceptable_inference_engine_release=acceptable_inference_engine_release
    )
    model_object = create_model_object(model=model, verbose=verbose)
    model_metadata = create_model_metadata(model_name=model_name, model_author=model_author,
                                           activation_function=activation_function,
                                           using_skip_connections=using_skip_connections)
    model_input_normalization_metadata = create_data_normalization_metadata(data_length=input_size,
                                                                            intercept=intercept,
                                                                            slope=slope,
                                                                            layer_label="inputs")
    model_output_normalization_metadata = create_data_normalization_metadata(data_length=output_size,
                                                                             intercept=intercept,
                                                                             slope=slope,
                                                                             layer_label="outputs")
    indent = "    "

    with open(nexport.append_extension(filename=filename, extension="json"), "w") as outfile:
        outfile.write("{\n")
        if include_metadata:
            # write the acceptable inference engine release
            for key in model_inference_engine_release_metadata:
                outfile.write(f"{indent}\"{key}\": \"{model_inference_engine_release_metadata[key]}\",\n")

            outfile.write(f"{indent}\"metadata\": " + "{\n")

            for d, data in enumerate(model_metadata.keys()):
                if type(model_metadata[data]) is str:
                    outfile.write(f"{indent}{indent}\"{data}\": \"{model_metadata[data]}\"")
                elif type(model_metadata[data]) is bool:
                    outfile.write(f"{indent}{indent}\"{data}\": {str(model_metadata[data]).lower()}")
                else:
                    outfile.write(f"{indent}{indent}\"{data}\": null")
                if d < len(model_metadata.keys()) - 1:
                    outfile.write(",\n")
                else:
                    outfile.write("\n")
            outfile.write(f"{indent}" + "},\n")

            input_normalization_key = "inputs_map"
            output_normalization_key = "outputs_map"

            outfile.write(f"{indent}\"{input_normalization_key}\": " + "{\n")
            for d, data in enumerate(model_input_normalization_metadata.keys()):
                if type(model_input_normalization_metadata[data]) is str:
                    outfile.write(f"{indent}{indent}\"{data}\": \"{model_input_normalization_metadata[data]}\"")
                else:
                    outfile.write(f"{indent}{indent}\"{data}\": [")
                    for z, parameter in enumerate(model_input_normalization_metadata[data]):
                        if z < len(model_input_normalization_metadata[data]) - 1:
                            outfile.write(f"{parameter}, ")
                        else:
                            outfile.write(f"{parameter}")
                    outfile.write(f"]")

                if d < len(model_input_normalization_metadata.keys()) - 1:
                    outfile.write(f",\n")
                else:
                    outfile.write(f"\n")

            outfile.write(f"{indent}" + "},\n")

            outfile.write(f"{indent}\"{output_normalization_key}\": " + "{\n")
            for d, data in enumerate(model_output_normalization_metadata.keys()):
                if type(model_output_normalization_metadata[data]) is str:
                    outfile.write(f"{indent}{indent}\"{data}\": \"{model_output_normalization_metadata[data]}\"")
                else:
                    outfile.write(f"{indent}{indent}\"{data}\": [")
                    for z, parameter in enumerate(model_output_normalization_metadata[data]):
                        if z < len(model_output_normalization_metadata[data]) - 1:
                            outfile.write(f"{parameter}, ")
                        else:
                            outfile.write(f"{parameter}")
                    outfile.write(f"]")

                if d < len(model_output_normalization_metadata.keys()) - 1:
                    outfile.write(f",\n")
                else:
                    outfile.write(f"\n")

            outfile.write(f"{indent}" + "},\n")

        for layer_type in model_object.keys():
            if layer_type == "hidden_layers":
                outfile.write(f"{indent}\"{layer_type}\": [\n")
                for x, layer in enumerate(model_object[layer_type]):
                    outfile.write(f"{indent}{indent}[\n")
                    for y, neuron in enumerate(layer):
                        outfile.write(f"{indent}{indent}{indent}" + "{\n")
                        for param_type in neuron.keys():
                            if param_type == "weights":
                                outfile.write(f"{indent}{indent}{indent}{indent}\"{param_type}\": [")
                                for z, parameter in enumerate(neuron[param_type]):
                                    if z < len(neuron[param_type]) - 1:
                                        outfile.write(f"{parameter}, ")  # weight array
                                    else:
                                        outfile.write(f"{parameter}")  # last weight array element
                                outfile.write(f"],\n")
                            if param_type == "bias":
                                outfile.write(
                                    f"{indent}{indent}{indent}{indent}\"{param_type}\": {neuron[param_type]}\n")
                        if y < len(layer) - 1:
                            outfile.write(f"{indent}{indent}{indent}" + "},\n")  # end of neuron
                        else:
                            outfile.write(f"{indent}{indent}{indent}" + "}\n")  # last neuron array element
                    if x < len(model_object[layer_type]) - 1:
                        outfile.write(f"{indent}{indent}],\n")  # end of layer
                    else:
                        outfile.write(f"{indent}{indent}]\n")  # last layer array element
                outfile.write(f"{indent}],\n")  # end of hidden layer array
            if layer_type == "output_layer":
                outfile.write(f"{indent}\"{layer_type}\": [\n")
                for n, neuron in enumerate(model_object[layer_type]):
                    outfile.write(f"{indent}{indent}" + "{\n")
                    for param_type in neuron.keys():
                        if param_type == "weights":
                            outfile.write(f"{indent}{indent}{indent}\"{param_type}\": [")
                            for z, parameter in enumerate(neuron[param_type]):
                                if z < len(neuron[param_type]) - 1:
                                    outfile.write(f"{parameter}, ")  # weight array
                                else:
                                    outfile.write(f"{parameter}")  # last weight array alement
                            outfile.write(f"],\n")
                        if param_type == "bias":
                            outfile.write(f"{indent}{indent}{indent}\"{param_type}\": {neuron[param_type]}\n")
                    if n < len(model_object[layer_type]) - 1:
                        outfile.write(f"{indent}{indent}" + "},\n")
                    else:
                        outfile.write(f"{indent}{indent}" + "}\n")
                outfile.write(f"{indent}]\n")
        outfile.write("}")

    t2 = t.time()
    time = t2 - t1

    if verbose >= 1:  # if verbose set to at least 1
        print(
            f"{c.CYAN}Exported model to {c.LIGHTCYAN}'{nexport.append_extension(filename=filename, extension='json')}'{c.CYAN}!{c.DEFAULT}")
    if verbose >= 2:  # if verbose set to at least 2
        print(f"{c.MAGENTA}    Time taken: {c.LIGHTMAGENTA}{round(time, 2)}{c.MAGENTA}s{c.DEFAULT}")
