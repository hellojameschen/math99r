#!/usr/bin/python
import pygame
import argparse as ap
import sys
from time import sleep
from cmath import *

def round(p):
    x, y = p
    return (int(x), int(y))

p = ap.ArgumentParser(description = "Replicate drawing using Fourier Transform")
p.add_argument("-n", default = 10, type = int, help = "number of circles to use (default is 10)")
p.add_argument("-m", "--mode", default = "loop", choices = ["loop", "increase"], help = "whether the number of circles should stay constant or increase after each loop (default is 'loop')")
p.add_argument("-d", "--decay", default = 0.6, type = float, help = "from 0 to 1, how strongly should the track brightness decay (default is 0.6)")
p.add_argument("-t", "--trace", action="store_true", help = "if set the original handdrawn trace will be left visible while the program replicates it")
args = p.parse_args()

decay = max(0, min(1, args.decay))

w, h = 1920, 1080

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((w, h))

white = pygame.Color(220, 220, 220)
yellow = pygame.Color(255, 255, 0)
grey = pygame.Color(120, 120, 120)
dark_grey = pygame.Color(90, 90, 90)
black = pygame.Color(30, 30, 30)

def draw_grid():
    screen.fill(black)
    pygame.draw.line(screen, grey, (w//2, 0), (w//2, h))
    for i in range(50, w//2, 50):
        pygame.draw.line(screen, dark_grey, (w//2+i, 0), (w//2+i, h))
        pygame.draw.line(screen, dark_grey, (w//2-i, 0), (w//2-i, h))
    pygame.draw.line(screen, grey, (0, h//2), (w, h//2))
    for i in range(50, h//2, 50):
        pygame.draw.line(screen, dark_grey, (0, h//2+i), (w, h//2+i))
        pygame.draw.line(screen, dark_grey, (0, h//2-i), (w, h//2-i))

draw_grid()
font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
start_text = font.render("press SPACE to start/stop drawing, ESC to exit", True, white)
screen.blit(start_text, ((w - start_text.get_width())//2, (h - start_text.get_height())//2))
pygame.display.update()

wait = True
while wait:
    sleep(0.005)
    for e in pygame.event.get():
        if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
            wait = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            sys.exit()

print("recording track")

track = [pygame.mouse.get_pos()]
wait = True
while wait:
    sleep(0.005)
    for e in pygame.event.get():
        if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
            wait = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            sys.exit()
    x0, y0 = track[-1]
    x1, y1 = pygame.mouse.get_pos()
    d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
    for i in range(2, int(d), 2):
        track.append((x0 + (x1-x0)*i/d, y0 + (y1-y0)*i/d))
    draw_grid()
    for p in track:
        screen.set_at(round(p), white)
    pygame.display.update()

print("processing track")

tl = len(track)
for i in range(tl):
    x, y = track[i]
    track[i] = (x-w//2, y-h//2)

ftrack = []
n = args.n
while True:
    print("drawing witn n = %d"%n)
    if ftrack == []:
        c = []
        for i in range(n, -n-1, -1):
            c.append(sum(exp(2*pi*1j*i*t/tl)*(track[t][0]+track[t][1]*1j) for t in range(tl))/tl)

    for t in range(tl):
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                sys.exit()
        # sleep(0.005)
        screen.fill(black)
        z = w//2 + h//2*1j
        # for i in range(2*n+1):
        for i in sum(zip(range(n+1, 2*n+1), range(n-1, -1, -1)), (n,)):
            old_z = z
            z += exp(2*pi*1j*(i-n)*t/tl)*c[i] 
            pygame.draw.line(screen, grey, (old_z.real, old_z.imag), (z.real, z.imag))
            r = ((old_z.real - z.real)**2 + (old_z.imag - z.imag)**2)**0.5
            if r > 1:
                pygame.draw.circle(screen, dark_grey, (int(old_z.real), int(old_z.imag)), int(r), 1)
        if len(ftrack) < tl:
            ftrack.append(z)

        #z = sum(exp(2*pi*1j*(i-n)*t/tl)*c[i] for i in range(2*n+1))
        if args.trace:
            for p in track:
                screen.set_at(round((p[0]+w//2, p[1]+h//2)), grey)
        for i in range(len(ftrack)):
            color = yellow[:]
            color = [*(int(color[j] * (1 - decay*((t - i + tl) % tl) / tl)) for j in range(3))]+[255]
            p = ftrack[i]
            screen.set_at((int(p.real), int(p.imag)), color)
        pygame.display.update()

    if args.mode == "increase":
        n += 1
        ftrack = []
