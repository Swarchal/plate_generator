# plate_generator

**Generate synthetic microtitre plate results with and without artefacts.**


## Installation


## Examples
`plate_generator` creates `plate` objects, which have three main attributes:
- `plate.data`: typical combination of values and any artefact
- `plate.effect`: the artefact present on the plate
- `plate.noise`: the values with *without* any artefact

<img src="/assets/edge_plate_data.png" height="250" align="right"/>

```python
import plate_generator
import matplotlib.pyplot as plt

edge_plate = plate_generator.edge_plate()

plt.imshow(edge_plate.data)
```


<img src="/assets/row_plate_data.png" height="250" align="right"/>

```python
row_plate = plate_generator.row_plate()

plt.imshow(row_plate.data)
```

Plates can be combined with mathematical operations (`+`, `-`, `*`, `\`, ...)


<img src="/assets/combined_plate_data.png" height="250" align="right"/>

```python
combined = row_plate + edge_plate

plt.imshow(combined.data)
```

The artefacts can be separated from the noise.

```python
plt.subplot(211)
plt.imshow(combined.effect)
plt.title("Artefact")
plt.subplot(212)
plt.imshow(combined.noise)
plt.imshow("Noise")
```

<img src="/assets/combined_separated.png" height="400"/>
