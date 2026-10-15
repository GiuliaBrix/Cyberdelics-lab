"""
my_deepdream.py - Addestra una CNN tua e usala per fare DeepDream.

Uso:
  1) Addestramento:
       python my_deepdream.py train --data data --epochs 25 --out my_net.keras

  2) Vedere i layer disponibili per il dream:
       python my_deepdream.py layers --model my_net.keras

  3) Dream:
       python my_deepdream.py dream --model my_net.keras --image foto.jpg \
           --layers conv6_act,conv8_act --out dream.png

Requisiti: pip install tensorflow pillow numpy
"""
import argparse

import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------------------------------------------------
# 1) LA RETE
# ----------------------------------------------------------------------
def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU. L'uscita (name + '_act') e' il layer da 'sognare'."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name)(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    x = layers.Activation("relu", name=name + "_act")(x)
    return x


def build_net(num_classes=10):
    # Input a dimensione libera + GlobalAveragePooling alla fine:
    # cosi' la rete accetta immagini di qualsiasi dimensione nel dream.
    inp = layers.Input(shape=(None, None, 3))
    x = conv_block(inp, 32, "conv1")
    x = conv_block(x, 32, "conv2")
    x = layers.MaxPooling2D()(x)

    x = conv_block(x, 64, "conv3")
    x = conv_block(x, 64, "conv4")
    x = layers.MaxPooling2D()(x)

    x = conv_block(x, 128, "conv5")
    x = conv_block(x, 128, "conv6")
    x = layers.MaxPooling2D()(x)

    x = conv_block(x, 256, "conv7")
    x = conv_block(x, 256, "conv8")

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="my_net")


# ----------------------------------------------------------------------
# 2) ADDESTRAMENTO da cartelle: data/eyes, data/fractals, data/kaleidoscope
# ----------------------------------------------------------------------
def train(args):
    size = (args.size, args.size)
    common = dict(
        directory=args.data, image_size=size, batch_size=args.batch,
        validation_split=0.2, seed=1337, label_mode="int",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names
    print("Classi:", class_names)

    # Preprocessing: solo scala [0, 1]. Nel dream useremo lo stesso range,
    # quindi non serve alcun "deprocess".
    def scale(img, label):
        return img / 255.0, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return tf.clip_by_value(img, 0.0, 1.0), label

    train_ds = (train_ds.map(scale, tf.data.AUTOTUNE)
                .map(augment, tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = val_ds.map(scale, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    net = build_net(num_classes=len(class_names))
    net.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    net.summary()
    net.fit(train_ds, epochs=args.epochs, validation_data=val_ds)
    net.save(args.out)
    print(f"Modello salvato in {args.out}")


# ----------------------------------------------------------------------
# 3) DEEPDREAM
# ----------------------------------------------------------------------
def load_image(path, max_dim):
    img = PIL.Image.open(path).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    return np.array(img, dtype=np.float32) / 255.0


def save_image(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    PIL.Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def make_stepper(net, layer_names):
    """Crea la funzione che fa UN passo di gradient ascent sull'immagine."""
    outputs = [net.get_layer(n).output for n in layer_names]
    extractor = tf.keras.Model(inputs=net.input, outputs=outputs)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def step(img, lr):
        with tf.GradientTape() as tape:
            tape.watch(img)
            acts = extractor(img[tf.newaxis], training=False)
            if len(layer_names) == 1:
                acts = [acts]
            # La loss e' la media delle attivazioni: la massimizziamo.
            loss = tf.add_n([tf.reduce_mean(a) for a in acts])
        grads = tape.gradient(loss, img)
        grads /= tf.math.reduce_std(grads) + 1e-8  # normalizza il gradiente
        img = tf.clip_by_value(img + lr * grads, 0.0, 1.0)
        return loss, img

    return step


def deepdream(net, img, layer_names, steps, lr, octave_range, octave_scale):
    step = make_stepper(net, layer_names)
    img = tf.convert_to_tensor(img)
    base_shape = tf.cast(tf.shape(img)[:2], tf.float32)

    for n in octave_range:
        new_shape = tf.cast(base_shape * (octave_scale ** n), tf.int32)
        img = tf.clip_by_value(tf.image.resize(img, new_shape), 0.0, 1.0)
        for i in range(steps):
            loss, img = step(img, tf.constant(lr, dtype=tf.float32))
        print(f"ottava {n:+d} -> {tuple(new_shape.numpy())}, loss = {loss.numpy():.4f}")

    out = tf.image.resize(img, tf.cast(base_shape, tf.int32))
    return tf.clip_by_value(out, 0.0, 1.0).numpy()


def run_dream(args):
    net = tf.keras.models.load_model(args.model)
    img = load_image(args.image, args.max_dim)
    layer_names = [n.strip() for n in args.layers.split(",")]
    octave_range = range(-args.octaves, args.octaves + 1)
    out = deepdream(net, img, layer_names, args.steps, args.lr,
                    octave_range, args.octave_scale)
    save_image(out, args.out)
    print(f"Salvato in {args.out}")


def list_layers(args):
    net = tf.keras.models.load_model(args.model)
    for layer in net.layers:
        if layer.name.endswith("_act"):
            print(layer.name, layer.output.shape)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--size", type=int, default=128)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--out", default="my_net.keras")
    t.set_defaults(func=train)

    l = sub.add_parser("layers")
    l.add_argument("--model", default="my_net.keras")
    l.set_defaults(func=list_layers)

    d = sub.add_parser("dream")
    d.add_argument("--model", default="my_net.keras")
    d.add_argument("--image", required=True)
    d.add_argument("--layers", default="conv6_act,conv8_act")
    d.add_argument("--out", default="dream.png")
    d.add_argument("--steps", type=int, default=30, help="passi per ottava")
    d.add_argument("--lr", type=float, default=0.01)
    d.add_argument("--octaves", type=int, default=2, help="ottave da -N a +N")
    d.add_argument("--octave-scale", type=float, default=1.3)
    d.add_argument("--max-dim", type=int, default=400)
    d.set_defaults(func=run_dream)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
