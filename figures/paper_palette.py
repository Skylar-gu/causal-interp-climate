"""One palette for every chart in the paper.

Light ground, restrained hues drawn from the Ida panels: the cool blues of the pressure
contours, a sage green, and the warm ochre at the hot end of the storm's own colourmap.
Hue is never the only channel -- marker shape, fill and dash carry the same distinction --
so the figures survive a grayscale print.
"""
BG    = "#ffffff"
INK   = "#1a1a1a"          # primary text and the strongest data
MUTED = "#5f5f5f"          # secondary text
FAINT = "#909090"          # tertiary text, annotations
GRIDC = "#e6e6e6"          # gridlines
BLUE   = "#2f6f8f"         # the measured signal
GREEN  = "#4e9078"         # a second real series
YELLOW = "#c9862b"         # accent, and the grid-locked class
GREY   = "#9b9b9b"         # matched controls
PALE   = "#d2d2d2"         # arms that could not be read
