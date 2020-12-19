import json
import os
from collections import OrderedDict

import numpy as np
import tensorflow.keras.backend as K
from keras.layers import Layer
from keras.models import Model


def _convert_1d_to_2d(num_units: int):
    # find divisors of num_units.
    divisors = []
    for i in range(1, num_units + 1):
        q = num_units / i
        if int(q) == q:
            divisors.append(i)
    divisors = list(reversed(divisors))
    pairs = []
    for d in divisors:
        for e in divisors[1:]:
            if d * e == num_units:
                pairs.append((d, e))
    if len(pairs) == 0:
        return num_units, 1
    # square x*y == rectangle x*y but minimizes x+y.
    close_to_square_id = int(np.argmin(np.sum(np.array(pairs), axis=1)))
    return pairs[close_to_square_id]


def n_(node, output_format_, nested=False):
    if isinstance(node, list):
        node_name = '_'.join([str(n.name) for n in node])
    else:
        node_name = str(node.name)
    if output_format_ == 'simple':
        if '/' in node_name:
            # This ensures that subnodes get properly named.
            tokens = node_name.split('/')
            if nested:
                return '/'.join(tokens[:-1])
            else:
                return tokens[0]
        elif ':' in node_name:
            return node_name.split(':')[0]
        else:
            return node_name
    return node_name


def _evaluate(model: Model, nodes_to_evaluate, x, y=None, auto_compile=False):
    if not model._is_compiled:
        # tensorflow.python.keras.applications.*
        applications_model_names = [
            'densenet',
            'efficientnet',
            'inception_resnet_v2',
            'inception_v3',
            'mobilenet',
            'mobilenet_v2',
            'nasnet',
            'resnet',
            'resnet_v2',
            'vgg16',
            'vgg19',
            'xception'
        ]
        if model.name in applications_model_names:
            print('Transfer learning detected. Model will be compiled with ("categorical_crossentropy", "adam").')
            print('If you want to change the default behaviour, then do in python:')
            print('model.name = ""')
            print('Then compile your model with whatever loss you want: https://keras.io/models/model/#compile.')
            print('If you want to get rid of this message, add this line before calling keract:')
            print('model.compile(loss="categorical_crossentropy", optimizer="adam")')
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            if auto_compile:
                model.compile(loss='mse', optimizer='adam')
            else:
                print('Please compile your model first! https://keras.io/models/model/#compile.')
                print('If you only care about the activations (outputs of the layers), '
                      'then just compile your model like that:')
                print('model.compile(loss="mse", optimizer="adam")')
                raise Exception('Compilation of the model required.')

    def eval_fn(k_inputs):
        try:
            return K.function(k_inputs, nodes_to_evaluate)(model._standardize_user_data(x, y))
        except AttributeError:  # one way to avoid forcing non eager mode.
            if y is None:  # tf 2.3.0 upgrade compatibility.
                return K.function(k_inputs, nodes_to_evaluate)(x)
            return K.function(k_inputs, nodes_to_evaluate)((x, y))  # although works.
        except ValueError as e:
            raise e

    try:
        return eval_fn(model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    except Exception:
        return eval_fn(model._feed_inputs)


def _get_nodes(module, output_format, nested=False, layer_names=[]):
    is_model_or_layer = isinstance(module, Model) or isinstance(module, Layer)
    has_layers = hasattr(module, '_layers') and bool(module._layers)
    assert is_model_or_layer, 'Not a model or layer!'

    def output(u):
        try:
            return u.output
        except AttributeError:  # for example Sequential. After tf2.3.
            return u.outbound_nodes[0].outputs

    try:
        module_name = n_(module.output, output_format_=output_format, nested=nested)
    except AttributeError:  # for example Sequential. After tf2.3.
        module_name = module.name

    if has_layers:
        node_dict = OrderedDict()
        for m in module._layers:
            try:
                if isinstance(m, dict) and len(m) == 0:
                    continue
                key = n_(m.output, output_format_=output_format, nested=nested)
            except AttributeError:  # for example Sequential. After tf2.3.
                key = m.name
            if nested:
                nodes = _get_nodes(m, output_format,
                                   nested=nested,
                                   layer_names=layer_names)
            else:
                if bool(layer_names) and key in layer_names:
                    nodes = OrderedDict([(key, output(m))])
                elif not bool(layer_names):
                    nodes = OrderedDict([(key, output(m))])
                else:
                    nodes = OrderedDict()
            node_dict.update(nodes)
        return node_dict

    elif bool(layer_names) and module_name in layer_names:
        # print("1", module_name, module)
        return OrderedDict({module_name: module.output})

    elif not bool(layer_names):
        # print("2", module_name, module)
        return OrderedDict({module_name: module.output})

    else:
        # print("3", module_name, module)
        return OrderedDict()


def get_activations(model, x, layer_names=None, nodes_to_evaluate=None,
                    output_format='simple', nested=False, auto_compile=True):

    layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
    # print('Layer names:', layer_names)
    if nodes_to_evaluate is None:
        nodes = _get_nodes(model, output_format, layer_names=layer_names, nested=nested)
    else:
        if layer_names is not None:
            raise ValueError('Do not specify a [layer_name] with [nodes_to_evaluate]. It will not be used.')
        nodes = OrderedDict([(n_(node, 'full'), node) for node in nodes_to_evaluate])

    if len(nodes) == 0:
        if layer_names is not None:
            network_layers = ', '.join([layer.name for layer in model.layers])
            raise KeyError('Could not find a layer with name: [{}]. '
                           'Network layers are [{}]'.format(', '.join(layer_names), network_layers))
        else:
            raise ValueError('Nodes list is empty. Or maybe the model is empty.')

    # The placeholders are processed later (Inputs node in Keras). Due to a small bug in tensorflow.
    input_layer_outputs = []
    layer_outputs = OrderedDict()

    for key, node in nodes.items():
        if isinstance(node, list):
            for nod in node:
                if nod.op.type != 'Placeholder':  # no inputs please.
                    layer_outputs.update({key: node})
        else:
            if node.op.type != 'Placeholder':  # no inputs please.
                layer_outputs.update({key: node})
    if nodes_to_evaluate is None or (layer_names is not None) and \
            any([n.name in layer_names for n in model.inputs]):
        input_layer_outputs = list(model.inputs)

    if len(layer_outputs) > 0:
        activations = _evaluate(model, layer_outputs.values(), x, y=None, auto_compile=auto_compile)
    else:
        activations = {}

    def craft_output(output_format_):
        inputs = [x] if not isinstance(x, list) else x
        activations_inputs_dict = OrderedDict(
            zip([n_(output, output_format_) for output in input_layer_outputs], inputs))
        activations_dict = OrderedDict(zip(layer_outputs.keys(), activations))
        result_ = activations_inputs_dict.copy()
        result_.update(activations_dict)

        if output_format_ == 'numbered':
            result_ = OrderedDict([(i, v) for i, (k, v) in enumerate(result_.items())])
        return result_

    result = craft_output(output_format)
    if layer_names is not None:  # extra check.
        result = {k: v for k, v in result.items() if k in layer_names}
    if nodes_to_evaluate is not None and len(result) != len(nodes_to_evaluate):
        result = craft_output(output_format_='full')  # collision detected in the keys.

    return result