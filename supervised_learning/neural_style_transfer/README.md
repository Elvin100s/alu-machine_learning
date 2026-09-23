# Neural Style Transfer

Implementation of Neural Style Transfer (Gatys et al., *A Neural Algorithm of
Artistic Style*) with TensorFlow eager execution and a VGG19 feature
extractor.

## Requirements

* Ubuntu 16.04 LTS, python3 (3.5)
* numpy (1.15), tensorflow (1.12)
* pycodestyle (2.4)

## Concepts

* **Neural style transfer** — optimizing an image so that its deep content
  features match a content image while its feature correlations match a style
  image.
* **Gram matrix** — the matrix of correlations between the channels of a layer
  output; it captures style while discarding spatial arrangement.
* **Content cost** — mean squared error between the content layer output of
  the generated and content images.
* **Style cost** — sum over the style layers of the mean squared error between
  the gram matrices of the generated and style images, weighted evenly.
* **Variational cost** — total variation of the generated image, used to
  penalize high frequency pixel noise.
* **Gradient Tape** — records the forward pass so gradients of the total cost
  can be taken with respect to the generated image.

## Files

| File | Description |
| --- | --- |
| `0-neural_style.py` | `NST` class: constructor and `scale_image` |
| `1-neural_style.py` | adds `load_model` (VGG19 with average pooling) |
| `2-neural_style.py` | adds `gram_matrix` |
| `3-neural_style.py` | adds `generate_features` |
| `4-neural_style.py` | adds `layer_style_cost` |
| `5-neural_style.py` | adds `style_cost` |
| `6-neural_style.py` | adds `content_cost` |
| `7-neural_style.py` | adds `total_cost` |
| `8-neural_style.py` | adds `compute_grads` |
| `9-neural_style.py` | adds `generate_image` (Adam optimization) |
| `10-neural_style.py` | adds `variational_cost` and the `var` weight |

## Usage

```
$ ./10-main.py
```

`10-main.py` reads `starry_night.jpg` as the style image and
`golden_gate.jpg` as the content image, runs gradient descent, and saves the
stylized result.

## Author

Elvin Cyubahiro
