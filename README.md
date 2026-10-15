# Dream: eyes, fractals, kaleidoscopes

A small CNN trained from scratch to recognize three classes (**eyes**, **fractals**, **kaleidoscopes**), then used in **DeepDream** style. Instead of an ImageNet network doing the dreaming, the network only knows these three worlds.

## How it works

1. `make_dataset.py` procedurally generates the `fractals` class (Julia, Mandelbrot, Burning Ship) and the `kaleidoscope` class (angular folding of random shapes).
2. You add your own eye photos to `data/eyes/`.
3. `my_deepdream.py` trains the network, then applies gradient ascent on the chosen layers across multiple scales (octaves).

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install tensorflow pillow numpy
```

## Usage

```bash
# 1) generate fractals and kaleidoscopes
python make_dataset.py --out data --n 2000 --size 128

# 2) put your eye photos in data/eyes/

# 3) train
python my_deepdream.py train --data data --epochs 25 --out my_net.keras

# 4) list the available layers
python my_deepdream.py layers --model my_net.keras

# 5) dream
python my_deepdream.py dream --model my_net.keras --image photo.jpg \
    --layers conv6_act,conv8_act --out dream.png
```

For more "hybrid" dreams, `make_dataset.py --kaleido-from data/eyes` builds part of the kaleidoscopes from your eye photos.

## Dream parameters

| Parameter | Default | Effect |
|---|---|---|
| `--layers` | `conv6_act,conv8_act` | deeper layers = abstract shapes, shallower layers = textures |
| `--lr` | 0.01 | raise it if little changes, lower it if the image burns out |
| `--steps` | 30 | gradient ascent steps per octave |
| `--octaves` | 2 | octaves from -N to +N |
| `--octave-scale` | 1.3 | scale ratio between octaves |
| `--max-dim` | 400 | maximum size of the input image |

## Notes

- The eye dataset is **not included**: check the license of the images you use, especially if they contain faces.
- `data/` and the trained models (`*.keras`) are excluded from the repository via `.gitignore`.
- The network is trained on 128x128 images: for better results, use input images that are not too large, or increase `--octave-scale`.

## Examples

Put your best results in the `examples/` folder.
