#!/usr/bin/env python3
# Utility functions for nonflowers generator

import math
import random
import time
import os
import base64
import json
from PIL import Image, ImageDraw

# Constants
PI = math.pi
rad2deg = 180 / PI
deg2rad = PI / 180

class PrngClass:
    """Seedable pseudo-random number generator similar to the JS implementation"""
    
    def __init__(self):
        self.s = 1234
        self.p = 999979
        self.q = 999983
        self.m = self.p * self.q
    
    def hash(self, x):
        """Hash function for seed generation"""
        y = base64.b64encode(json.dumps(x).encode()).decode()
        z = 0
        for i, c in enumerate(y):
            z += ord(c) * (128 ** i)
        return z
    
    def seed(self, x=None):
        """Set the random seed"""
        if x is None:
            x = int(time.time() * 1000)
        y = 0
        z = 0
        
        def redo():
            nonlocal y, z
            y = (self.hash(x) + z) % self.m
            z += 1
        
        while y % self.p == 0 or y % self.q == 0 or y == 0 or y == 1:
            redo()
        
        self.s = y
        print(f"int seed: {self.s}")
        for _ in range(10):
            self.next()
    
    def next(self):
        """Get next random number in [0, 1)"""
        self.s = (self.s * self.s) % self.m
        return self.s / self.m

# Initialize PRNG
Prng = PrngClass()

# Override random.random with our seeded PRNG
random_original = random.random
def seeded_random():
    return Prng.next()
random.random = seeded_random

def seed(x=None):
    """Seed the random number generator"""
    Prng.seed(x)

# Perlin noise implementation
class NoiseClass:
    """Perlin noise implementation similar to the JS version"""
    
    def __init__(self):
        self.perlin = None
        self.perlin_octaves = 4
        self.perlin_amp_falloff = 0.5
    
    def scaled_cosine(self, i):
        """Helper function for noise generation"""
        return 0.5 * (1.0 - math.cos(i * PI))
    
    def noise(self, x, y=0, z=0):
        """Generate perlin noise value"""
        if self.perlin is None:
            self.perlin = [random.random() for _ in range(4096)]
        
        PERLIN_YWRAPB = 4
        PERLIN_YWRAP = 1 << PERLIN_YWRAPB
        PERLIN_ZWRAPB = 8
        PERLIN_ZWRAP = 1 << PERLIN_ZWRAPB
        PERLIN_SIZE = 4095
        
        x = abs(x)
        y = abs(y)
        z = abs(z)
        
        xi = math.floor(x)
        yi = math.floor(y)
        zi = math.floor(z)
        xf = x - xi
        yf = y - yi
        zf = z - zi
        
        r = 0
        ampl = 0.5
        
        for o in range(self.perlin_octaves):
            of = xi + (yi << PERLIN_YWRAPB) + (zi << PERLIN_ZWRAPB)
            
            rxf = self.scaled_cosine(xf)
            ryf = self.scaled_cosine(yf)
            
            n1 = self.perlin[of & PERLIN_SIZE]
            n1 += rxf * (self.perlin[(of + 1) & PERLIN_SIZE] - n1)
            n2 = self.perlin[(of + PERLIN_YWRAP) & PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + PERLIN_YWRAP + 1) & PERLIN_SIZE] - n2)
            n1 += ryf * (n2 - n1)
            
            of += PERLIN_ZWRAP
            n2 = self.perlin[of & PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + 1) & PERLIN_SIZE] - n2)
            n3 = self.perlin[(of + PERLIN_YWRAP) & PERLIN_SIZE]
            n3 += rxf * (self.perlin[(of + PERLIN_YWRAP + 1) & PERLIN_SIZE] - n3)
            n2 += ryf * (n3 - n2)
            
            n1 += self.scaled_cosine(zf) * (n2 - n1)
            
            r += n1 * ampl
            ampl *= self.perlin_amp_falloff
            xi <<= 1
            xf *= 2
            yi <<= 1
            yf *= 2
            zi <<= 1
            zf *= 2
            
            if xf >= 1.0:
                xi += 1
                xf -= 1
            if yf >= 1.0:
                yi += 1
                yf -= 1
            if zf >= 1.0:
                zi += 1
                zf -= 1
        
        return r
    
    def noiseSeed(self, seed):
        """Seed the noise function"""
        class Lcg:
            def __init__(self):
                self.m = 4294967296
                self.a = 1664525
                self.c = 1013904223
                self.seed = 0
                self.z = 0
            
            def setSeed(self, val):
                if val is None:
                    val = random.random() * self.m
                self.z = self.seed = val & 0xFFFFFFFF
            
            def getSeed(self):
                return self.seed
            
            def rand(self):
                self.z = (self.a * self.z + self.c) % self.m
                return self.z / self.m
        
        lcg = Lcg()
        lcg.setSeed(seed)
        self.perlin = [lcg.rand() for _ in range(4096)]

# Initialize Noise
Noise = NoiseClass()

# Utility functions
def distance(p0, p1):
    """Distance between 2 coordinates in 2D"""
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def mapval(value, istart, istop, ostart, ostop):
    """Map float from one range to another"""
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

def rand_choice(arr):
    """Random element from array"""
    return arr[math.floor(len(arr) * random.random())]

def norm_rand(m, M):
    """Normalized random number"""
    return mapval(random.random(), 0, 1, m, M)

def wtrand(func):
    """Weighted randomness"""
    x = random.random()
    y = random.random()
    if y < func(x):
        return x
    else:
        return wtrand(func)

def rand_gaussian():
    """Gaussian randomness"""
    return wtrand(lambda x: math.pow(math.e, -24 * math.pow(x - 0.5, 2))) * 2 - 1

def sigmoid(x, k=10):
    """Sigmoid curve"""
    return 1 / (1 + math.exp(-k * (x - 0.5)))

def bean(x):
    """Pseudo bean curve"""
    return math.pow(0.25 - math.pow(x - 0.5, 2), 0.5) * (2.6 + 2.4 * math.pow(x, 1.5)) * 0.54

def squircle(r, a):
    """Interpolate between square and circle"""
    def func(th):
        while th > PI / 2:
            th -= PI / 2
        while th < 0:
            th += PI / 2
        return r * math.pow(1 / (math.pow(math.cos(th), a) + math.pow(math.sin(th), a)), 1 / a)
    return func

def mid_pt(points):
    """Mid-point of an array of points"""
    result = [0, 0, 0]
    for p in points:
        result[0] += p[0] / len(points)
        result[1] += p[1] / len(points)
        result[2] += p[2] / len(points)
    return result

def bezmh(P, w=1):
    """Rational bezier curve"""
    if len(P) == 2:
        P = [P[0], mid_pt([P[0], P[1]]), P[1]]
    
    plist = []
    for j in range(len(P) - 2):
        if j == 0:
            p0 = P[j]
        else:
            p0 = mid_pt([P[j], P[j + 1]])
        
        p1 = P[j + 1]
        
        if j == len(P) - 3:
            p2 = P[j + 2]
        else:
            p2 = mid_pt([P[j + 1], P[j + 2]])
        
        pl = 20
        for i in range(pl + (1 if j == len(P) - 3 else 0)):
            t = i / pl
            u = (math.pow(1 - t, 2) + 2 * t * (1 - t) * w + t * t)
            plist.append([
                (math.pow(1 - t, 2) * p0[0] + 2 * t * (1 - t) * p1[0] * w + t * t * p2[0]) / u,
                (math.pow(1 - t, 2) * p0[1] + 2 * t * (1 - t) * p1[1] * w + t * t * p2[1]) / u,
                (math.pow(1 - t, 2) * p0[2] + 2 * t * (1 - t) * p1[2] * w + t * t * p2[2]) / u
            ])
    
    return plist

# 3D vector operations
class V3:
    """Tools for vectors in 3d"""
    
    def __init__(self):
        self.forward = [0, 0, 1]
        self.up = [0, 1, 0]
        self.right = [1, 0, 0]
        self.zero = [0, 0, 0]
    
    def rotvec(self, vec, axis, th):
        """Rotate a vector around an axis"""
        l, m, n = axis
        x, y, z = vec
        costh, sinth = math.cos(th), math.sin(th)
        
        mat = {}
        mat[(1, 1)] = l * l * (1 - costh) + costh
        mat[(1, 2)] = m * l * (1 - costh) - n * sinth
        mat[(1, 3)] = n * l * (1 - costh) + m * sinth
        
        mat[(2, 1)] = l * m * (1 - costh) + n * sinth
        mat[(2, 2)] = m * m * (1 - costh) + costh
        mat[(2, 3)] = n * m * (1 - costh) - l * sinth
        
        mat[(3, 1)] = l * n * (1 - costh) - m * sinth
        mat[(3, 2)] = m * n * (1 - costh) + l * sinth
        mat[(3, 3)] = n * n * (1 - costh) + costh
        
        return [
            x * mat[(1, 1)] + y * mat[(1, 2)] + z * mat[(1, 3)],
            x * mat[(2, 1)] + y * mat[(2, 2)] + z * mat[(2, 3)],
            x * mat[(3, 1)] + y * mat[(3, 2)] + z * mat[(3, 3)]
        ]
    
    def roteuler(self, vec, rot):
        """Apply Euler rotations to a vector"""
        result = vec.copy()
        if rot[2] != 0:
            result = self.rotvec(result, self.forward, rot[2])
        if rot[0] != 0:
            result = self.rotvec(result, self.right, rot[0])
        if rot[1] != 0:
            result = self.rotvec(result, self.up, rot[1])
        return result
    
    def scale(self, vec, p):
        """Scale a vector"""
        return [vec[0] * p, vec[1] * p, vec[2] * p]
    
    def copy(self, v0):
        """Copy a vector"""
        return [v0[0], v0[1], v0[2]]
    
    def add(self, v0, v):
        """Add vectors"""
        return [v0[0] + v[0], v0[1] + v[1], v0[2] + v[2]]
    
    def subtract(self, v0, v):
        """Subtract vectors"""
        return [v0[0] - v[0], v0[1] - v[1], v0[2] - v[2]]
    
    def mag(self, v):
        """Magnitude of a vector"""
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    
    def normalize(self, v):
        """Normalize a vector"""
        m = self.mag(v)
        if m == 0:
            return [0, 0, 0]
        p = 1 / m
        return [v[0] * p, v[1] * p, v[2] * p]
    
    def dot(self, u, v):
        """Dot product"""
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    
    def cross(self, u, v):
        """Cross product"""
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
        ]
    
    def angcos(self, u, v):
        """Cosine of angle between vectors"""
        return self.dot(u, v) / (self.mag(u) * self.mag(v) + 1e-10)
    
    def ang(self, u, v):
        """Angle between vectors"""
        ac = self.angcos(u, v)
        # Clamp to -1, 1 to avoid numerical errors
        ac = max(-1.0, min(1.0, ac))
        return math.acos(ac)
    
    def toeuler(self, v0):
        """Convert vector to euler angles"""
        ep = 5
        ma = 2 * PI
        mr = [0, 0, 0]
        
        for x in range(-180, 180, ep):
            for y in range(-90, 90, ep):
                r = [math.radians(x), math.radians(y), 0]
                v = self.roteuler([0, 0, 1], r)
                a = self.ang(v0, v)
                if a < math.radians(ep):
                    return r
                if a < ma:
                    ma = a
                    mr = r
        return mr
    
    def lerp(self, u, v, p):
        """Linear interpolation between vectors"""
        # Ensure we have 3D vectors by padding with zeros if needed
        u_3d = u + [0] * (3 - len(u)) if len(u) < 3 else u
        v_3d = v + [0] * (3 - len(v)) if len(v) < 3 else v
        
        return [
            u_3d[0] * (1 - p) + v_3d[0] * p,
            u_3d[1] * (1 - p) + v_3d[1] * p,
            u_3d[2] * (1 - p) + v_3d[2] * p
        ]

# Initialize v3
v3 = V3()

# Color functions
def rgba(r, g=None, b=None, a=1.0):
    """RGBA to RGBA tuple"""
    if g is None:
        g = r
    if b is None:
        b = g
    return (int(r), int(g), int(b), int(a * 255))

def hsv(h, s, v, a=1.0):
    """HSV to RGBA tuple"""
    h = (h % 360) / 60
    c = v * s
    x = c * (1 - abs((h % 2) - 1))
    m = v - c
    
    if 0 <= h < 1:
        r, g, b = c, x, 0
    elif 1 <= h < 2:
        r, g, b = x, c, 0
    elif 2 <= h < 3:
        r, g, b = 0, c, x
    elif 3 <= h < 4:
        r, g, b = 0, x, c
    elif 4 <= h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return rgba((r + m) * 255, (g + m) * 255, (b + m) * 255, a)

def lerp_hue(h0, h1, p):
    """Lerp hue wrapping around 360 degs"""
    methods = [
        [abs(h1 - h0), mapval(p, 0, 1, h0, h1)],
        [abs(h1 + 360 - h0), mapval(p, 0, 1, h0, h1 + 360)],
        [abs(h1 - 360 - h0), mapval(p, 0, 1, h0, h1 - 360)]
    ]
    methods.sort(key=lambda x: x[0])
    return (methods[0][1] + 720) % 360

def grot(P, ind):
    """Get rotation at given index of a poly-line"""
    p1 = P[ind]
    p2 = P[ind - 1]
    
    # Ensure we have 3D vectors
    p1_3d = p1 + [0] * (3 - len(p1)) if len(p1) < 3 else p1
    p2_3d = p2 + [0] * (3 - len(p2)) if len(p2) < 3 else p2
    
    d = v3.subtract(p1_3d, p2_3d)
    return v3.toeuler(d)

# Drawing functions
class Layer:
    """Canvas context operations"""
    
    @staticmethod
    def empty(w, h=None):
        """Create an empty image with alpha channel"""
        if h is None:
            h = w
        return Image.new('RGBA', (w, h), (0, 0, 0, 0))
    
    @staticmethod
    def draw(img):
        """Get a draw context for an image"""
        return ImageDraw.Draw(img)
    
    @staticmethod
    def blit(img0, img1, args=None):
        """Composite one image onto another"""
        if args is None:
            args = {}
        
        ble = args.get('ble', 'normal')
        xof = args.get('xof', 0)
        yof = args.get('yof', 0)
        
        # Different blending modes
        if ble == 'multiply':
            img0.paste(img1, (int(xof), int(yof)), mask=img1)
        else:  # normal
            img0.paste(img1, (int(xof), int(yof)), mask=img1)
    
    @staticmethod
    def filter(img, f):
        """Apply a filter function to an image"""
        pixels = img.load()
        width, height = img.size
        
        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                r1, g1, b1, a1 = f(x, y, r, g, b, a)
                pixels[x, y] = (int(r1), int(g1), int(b1), int(a1))
    
    @staticmethod
    def border(img, f):
        """Apply border to an image"""
        pixels = img.load()
        width, height = img.size
        
        for x in range(width):
            for y in range(height):
                nx = (x / width - 0.5) * 2
                ny = (y / height - 0.5) * 2
                theta = math.atan2(ny, nx)
                r_ = distance([nx, ny], [0, 0])
                rr_ = f(theta)
                
                if r_ > rr_:
                    pixels[x, y] = (0, 0, 0, 0)
    
    @staticmethod
    def bound(img):
        """Find the dirty region of an image"""
        width, height = img.size
        pixels = img.load()
        
        xmin = width
        xmax = 0
        ymin = height
        ymax = 0
        
        for x in range(width):
            for y in range(height):
                _, _, _, a = pixels[x, y]
                if a > 0.001:
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
                    ymin = min(ymin, y)
                    ymax = max(ymax, y)
        
        return {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax
        }

class Filter:
    """Collection of image filters"""
    
    @staticmethod
    def wispy(x, y, r, g, b, a):
        """Wispy filter effect"""
        n = Noise.noise(x * 0.2, y * 0.2)
        m = Noise.noise(x * 0.5, y * 0.5, 2)
        return [
            r,
            g * mapval(m, 0, 1, 0.95, 1),
            b * mapval(m, 0, 1, 0.9, 1),
            a * mapval(n, 0, 1, 0.5, 1)
        ]
    
    @staticmethod
    def fade(x, y, r, g, b, a):
        """Fade filter effect"""
        n = Noise.noise(x * 0.01, y * 0.01)
        return [
            r,
            g,
            b,
            a * max(0, min(mapval(n, 0, 1, 0, 1), 1))
        ]

def polygon(args):
    """Draw a polygon"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    pts = args.get('pts', [])
    col = args.get('col', (0, 0, 0, 255))
    fil = args.get('fil', True)
    str_ = args.get('str', not fil)
    
    # Convert pts to screen coordinates
    screen_pts = [(pt[0] + xof, pt[1] + yof) for pt in pts]
    
    # Create a draw object
    draw = ImageDraw.Draw(img)
    
    if fil and len(pts) > 2:
        draw.polygon(screen_pts, fill=col)
    
    if str_ and len(pts) > 1:
        draw.line(screen_pts + [screen_pts[0]] if len(pts) > 2 else screen_pts, fill=col, width=1)

def tubify(args):
    """Generate 2d tube shape from list of points"""
    pts = args.get('pts', [])
    wid = args.get('wid', lambda x: 10)
    
    vtxlist0 = []
    vtxlist1 = []
    
    for i in range(1, len(pts) - 1):
        w = wid(i / len(pts))
        a1 = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
        a2 = math.atan2(pts[i][1] - pts[i + 1][1], pts[i][0] - pts[i + 1][0])
        a = (a1 + a2) / 2
        if a < a2:
            a += PI
        
        vtxlist0.append([pts[i][0] + w * math.cos(a), pts[i][1] + w * math.sin(a)])
        vtxlist1.append([pts[i][0] - w * math.cos(a), pts[i][1] - w * math.sin(a)])
    
    l = len(pts) - 1
    a0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) - PI / 2
    a1 = math.atan2(pts[l][1] - pts[l - 1][1], pts[l][0] - pts[l - 1][0]) - PI / 2
    w0 = wid(0)
    w1 = wid(1)
    
    vtxlist0.insert(0, [pts[0][0] + w0 * math.cos(a0), pts[0][1] + w0 * math.sin(a0)])
    vtxlist1.insert(0, [pts[0][0] - w0 * math.cos(a0), pts[0][1] - w0 * math.sin(a0)])
    vtxlist0.append([pts[l][0] + w1 * math.cos(a1), pts[l][1] + w1 * math.sin(a1)])
    vtxlist1.append([pts[l][0] - w1 * math.cos(a1), pts[l][1] - w1 * math.sin(a1)])
    
    return [vtxlist0, vtxlist1]

def stroke(args):
    """Line work with weight function"""
    pts = args.get('pts', [])
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    col = args.get('col', (0, 0, 0, 255))
    wid = args.get('wid', lambda x: math.sin(x * PI) * mapval(Noise.noise(x * 10), 0, 1, 0.5, 1))
    
    vtxlist0, vtxlist1 = tubify({'pts': pts, 'wid': wid})
    
    # Combine the two vertex lists to form a polygon
    polygon_pts = vtxlist0 + list(reversed(vtxlist1))
    
    polygon({
        'img': img,
        'pts': polygon_pts,
        'fil': True,
        'col': col,
        'xof': xof,
        'yof': yof
    })
    
    return [vtxlist0, vtxlist1]

def paper(args=None):
    """Generate paper texture"""
    if args is None:
        args = {}
    
    col = args.get('col', [0.98, 0.91, 0.74])
    tex = args.get('tex', 20)
    spr = args.get('spr', 1)
    
    # Create a texture image
    reso = 512
    img = Image.new('RGBA', (reso, reso))
    pixels = img.load()
    
    for i in range(reso // 2 + 1):
        for j in range(reso // 2 + 1):
            c = (255 - Noise.noise(i * 0.1, j * 0.1) * tex * 0.5)
            c -= random.random() * tex
            
            r = (c * col[0])
            g = (c * col[1])
            b = (c * col[2])
            
            if Noise.noise(i * 0.04, j * 0.04, 2) * random.random() * spr > 0.7 or random.random() < 0.005 * spr:
                r = (c * 0.7)
                g = (c * 0.5)
                b = (c * 0.2)
            
            pixels[i, j] = (int(r), int(g), int(b), 255)
            pixels[reso - i - 1, j] = (int(r), int(g), int(b), 255)
            pixels[i, reso - j - 1] = (int(r), int(g), int(b), 255)
            pixels[reso - i - 1, reso - j - 1] = (int(r), int(g), int(b), 255)
    
    return img
