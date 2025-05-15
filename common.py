

from pygame.locals import *
from pygame import gfxdraw

def circle(surface,  color, x, y, radius):
    gfxdraw.aacircle(surface, x, y, radius, color)
    gfxdraw.filled_circle(surface, x, y, radius, color)
