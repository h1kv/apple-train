"""Convert the Teachable Machine TensorFlow.js export (apple-model/) to a Keras .h5 model.

Reads model.json + weights.bin, rebuilds the architecture (MobileNetV2 alpha=0.35
feature extractor + the trained dense head) and saves keras_model.h5.
"""
import json
import numpy as np
import tensorflow as tf

MODEL_DIR = "apple-model"


def load_tfjs_weights(model_dir):
    with open(f"{model_dir}/model.json") as f:
        model_json = json.load(f)
    raw = np.fromfile(f"{model_dir}/weights.bin", dtype=np.float32)
    weights = {}
    offset = 0
    for group in model_json["weightsManifest"]:
        for spec in group["weights"]:
            assert spec["dtype"] == "float32", f"unexpected dtype: {spec}"
            shape = spec["shape"]
            size = int(np.prod(shape)) if shape else 1
            weights[spec["name"]] = raw[offset:offset + size].reshape(shape)
            offset += size
    assert offset == raw.size, f"consumed {offset} of {raw.size} floats"
    return weights


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=0.35, include_top=False, weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(100, activation="relu", name="dense_Dense3")(x)
    out = tf.keras.layers.Dense(
        2, activation="softmax", use_bias=False, name="dense_Dense4"
    )(x)
    return tf.keras.Model(base.input, out)


def assign_weights(model, weights):
    assigned = 0
    for layer in model.layers:
        if not layer.weights:
            continue
        name = layer.name
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            vals = [weights[f"{name}/gamma"], weights[f"{name}/beta"],
                    weights[f"{name}/moving_mean"], weights[f"{name}/moving_variance"]]
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            vals = [weights[f"{name}/depthwise_kernel"]]
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            vals = [weights[f"{name}/kernel"]]
            if layer.use_bias:
                vals.append(weights[f"{name}/bias"])
        else:
            raise RuntimeError(f"unhandled layer with weights: {name} ({type(layer).__name__})")
        layer.set_weights(vals)
        assigned += len(vals)
    return assigned


def main():
    weights = load_tfjs_weights(MODEL_DIR)
    model = build_model()
    assigned = assign_weights(model, weights)
    print(f"assigned {assigned} of {len(weights)} weight tensors")
    assert assigned == len(weights), "some weights from weights.bin were not used"

    # sanity check: output must be a 2-class softmax distribution
    pred = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    assert pred.shape == (1, 2) and abs(pred.sum() - 1.0) < 1e-4, pred

    model.save("keras_model.h5")
    print("saved keras_model.h5, sample prediction:", pred[0])


if __name__ == "__main__":
    main()
