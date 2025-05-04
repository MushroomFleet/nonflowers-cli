import math
import random
from utils import (PI, v3, mapval, norm_rand, lerp_hue, hsv, rgba, 
                  polygon, tubify, stroke, Noise, grot)

def leaf(args):
    """Generate leaf-like structure"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    rot = args.get('rot', [PI / 2, 0, 0])
    len_ = args.get('len', 500)
    seg = args.get('seg', 40)
    wid = args.get('wid', lambda x: math.sin(x * PI) * 20)
    vei = args.get('vei', [1, 3])
    flo = args.get('flo', False)
    col = args.get('col', {
        'min': [90, 0.2, 0.3, 1],
        'max': [90, 0.1, 0.9, 1]
    })
    cof = args.get('cof', lambda x: x)
    ben = args.get('ben', lambda x: [norm_rand(-10, 10), 0, norm_rand(-5, 5)])
    
    disp = v3.zero
    crot = v3.zero
    P = [disp]
    ROT = [crot]
    L = [disp]
    R = [disp]
    
    def orient(v):
        return v3.roteuler(v, rot)
    
    for i in range(seg):
        p = i / (seg - 1)
        crot = v3.add(crot, v3.scale(ben(p), 1/seg))
        disp = v3.add(disp, orient(v3.roteuler([0, 0, len_/seg], crot)))
        w = wid(p)
        l = v3.add(disp, orient(v3.roteuler([-w, 0, 0], crot)))
        r = v3.add(disp, orient(v3.roteuler([w, 0, 0], crot)))
        
        if i > 0:
            v0 = v3.subtract(disp, L[-1])
            v1 = v3.subtract(l, disp)
            v2 = v3.cross(v0, v1)
            if not flo:
                lt = mapval(abs(v3.ang(v2, [0, -1, 0])), 0, PI, 1, 0)
            else:
                lt = p * norm_rand(0.95, 1)
            
            lt = cof(lt) or 0
            
            h = lerp_hue(col['min'][0], col['max'][0], lt)
            s = mapval(lt, 0, 1, col['min'][1], col['max'][1])
            v_ = mapval(lt, 0, 1, col['min'][2], col['max'][2])
            a = mapval(lt, 0, 1, col['min'][3], col['max'][3])
            
            polygon({
                'img': img,
                'pts': [l, L[-1], P[-1], disp],
                'xof': xof,
                'yof': yof,
                'fil': True,
                'str': True,
                'col': hsv(h, s, v_, a)
            })
            
            polygon({
                'img': img,
                'pts': [r, R[-1], P[-1], disp],
                'xof': xof,
                'yof': yof,
                'fil': True,
                'str': True,
                'col': hsv(h, s, v_, a)
            })
        
        P.append(disp)
        ROT.append(crot)
        L.append(l)
        R.append(r)
    
    if vei[0] == 1:
        for i in range(1, len(P)):
            for j in range(vei[1]):
                p = j / vei[1]
                
                p0 = v3.lerp(L[i-1], P[i-1], p)
                p1 = v3.lerp(L[i], P[i], p)
                
                q0 = v3.lerp(R[i-1], P[i-1], p)
                q1 = v3.lerp(R[i], P[i], p)
                
                polygon({
                    'img': img,
                    'pts': [p0, p1],
                    'xof': xof,
                    'yof': yof,
                    'fil': False,
                    'col': hsv(0, 0, 0, norm_rand(0.4, 0.9))
                })
                
                polygon({
                    'img': img,
                    'pts': [q0, q1],
                    'xof': xof,
                    'yof': yof,
                    'fil': False,
                    'col': hsv(0, 0, 0, norm_rand(0.4, 0.9))
                })
        
        stroke({
            'img': img,
            'pts': P,
            'xof': xof,
            'yof': yof,
            'col': rgba(0, 0, 0, 0.3)
        })
    
    elif vei[0] == 2:
        for i in range(1, len(P) - vei[1], vei[2]):
            polygon({
                'img': img,
                'pts': [P[i], L[i + vei[1]]],
                'xof': xof,
                'yof': yof,
                'fil': False,
                'col': hsv(0, 0, 0, norm_rand(0.4, 0.9))
            })
            
            polygon({
                'img': img,
                'pts': [P[i], R[i + vei[1]]],
                'xof': xof,
                'yof': yof,
                'fil': False,
                'col': hsv(0, 0, 0, norm_rand(0.4, 0.9))
            })
        
        stroke({
            'img': img,
            'pts': P,
            'xof': xof,
            'yof': yof,
            'col': rgba(0, 0, 0, 0.3)
        })
    
    stroke({
        'img': img,
        'pts': L,
        'xof': xof,
        'yof': yof,
        'col': rgba(120, 100, 0, 0.3)
    })
    
    stroke({
        'img': img,
        'pts': R,
        'xof': xof,
        'yof': yof,
        'col': rgba(120, 100, 0, 0.3)
    })
    
    return P

def stem(args):
    """Generate stem-like structure"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    rot = args.get('rot', [PI / 2, 0, 0])
    len_ = args.get('len', 400)
    seg = args.get('seg', 40)
    wid = args.get('wid', lambda x: 6)
    col = args.get('col', {
        'min': [250, 0.2, 0.4, 1],
        'max': [250, 0.3, 0.6, 1]
    })
    ben = args.get('ben', lambda x: [norm_rand(-10, 10), 0, norm_rand(-5, 5)])
    
    disp = v3.zero
    crot = v3.zero
    P = [disp]
    ROT = [crot]
    
    def orient(v):
        return v3.roteuler(v, rot)
    
    for i in range(seg):
        p = i / (seg - 1)
        crot = v3.add(crot, v3.scale(ben(p), 1/seg))
        disp = v3.add(disp, orient(v3.roteuler([0, 0, len_/seg], crot)))
        ROT.append(crot)
        P.append(disp)
    
    L, R = tubify({'pts': P, 'wid': wid})
    wseg = 4
    
    for i in range(1, len(P)):
        for j in range(1, wseg):
            m = (j - 1) / (wseg - 1)
            n = j / (wseg - 1)
            p = i / (len(P) - 1)
            
            p0 = v3.lerp(L[i-1], R[i-1], m)
            p1 = v3.lerp(L[i], R[i], m)
            
            p2 = v3.lerp(L[i-1], R[i-1], n)
            p3 = v3.lerp(L[i], R[i], n)
            
            lt = n / p if p != 0 else 0
            h = lerp_hue(col['min'][0], col['max'][0], lt) * mapval(Noise.noise(p*10, m*10, n*10), 0, 1, 0.5, 1)
            s = mapval(lt, 0, 1, col['max'][1], col['min'][1]) * mapval(Noise.noise(p*10, m*10, n*10), 0, 1, 0.5, 1)
            v_ = mapval(lt, 0, 1, col['min'][2], col['max'][2]) * mapval(Noise.noise(p*10, m*10, n*10), 0, 1, 0.5, 1)
            a = mapval(lt, 0, 1, col['min'][3], col['max'][3])
            
            polygon({
                'img': img,
                'pts': [p0, p1, p3, p2],
                'xof': xof,
                'yof': yof,
                'fil': True,
                'str': True,
                'col': hsv(h, s, v_, a)
            })
    
    stroke({
        'img': img,
        'pts': L,
        'xof': xof,
        'yof': yof,
        'col': rgba(0, 0, 0, 0.5)
    })
    
    stroke({
        'img': img,
        'pts': R,
        'xof': xof,
        'yof': yof,
        'col': rgba(0, 0, 0, 0.5)
    })
    
    return P

def branch(args):
    """Generate fractal-like branches"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    rot = args.get('rot', [PI / 2, 0, 0])
    len_ = args.get('len', 400)
    seg = args.get('seg', 40)
    wid = args.get('wid', 1)
    twi = args.get('twi', 5)
    col = args.get('col', {
        'min': [50, 0.2, 0.8, 1],
        'max': [50, 0.2, 0.8, 1]
    })
    dep = args.get('dep', 3)
    frk = args.get('frk', 4)
    
    jnt = []
    for i in range(twi):
        jnt.append([math.floor(random.random() * seg), norm_rand(-1, 1)])
    
    def jntdist(x):
        m = seg
        j = 0
        for i in range(len(jnt)):
            n = abs(x * seg - jnt[i][0])
            if n < m:
                m = n
                j = i
        return [m, jnt[j][1]]
    
    def wfun(x):
        m, j = jntdist(x)
        if m < 1:
            return wid * (3 + 5 * (1 - x))
        else:
            return wid * (2 + 7 * (1 - x) * mapval(Noise.noise(x * 10), 0, 1, 0.5, 1))
    
    def bfun(x):
        m, j = jntdist(x)
        if m < 1:
            return [0, j * 20, 0]
        else:
            return [0, norm_rand(-5, 5), 0]
    
    P = stem({
        'img': img,
        'xof': xof,
        'yof': yof,
        'rot': rot,
        'len': len_,
        'seg': seg,
        'wid': wfun,
        'col': col,
        'ben': bfun
    })
    
    child = []
    if dep > 0 and wid > 0.1:
        for i in range(int(frk * random.random())):
            ind = math.floor(norm_rand(1, len(P)))
            
            r = grot(P, ind)
            L = branch({
                'img': img,
                'xof': xof + P[ind][0],
                'yof': yof + P[ind][1],
                'rot': [
                    r[0] + norm_rand(-1, 1) * PI / 6,
                    r[1] + norm_rand(-1, 1) * PI / 6,
                    r[2] + norm_rand(-1, 1) * PI / 6
                ],
                'seg': seg,
                'len': len_ * norm_rand(0.4, 0.6),
                'wid': wid * norm_rand(0.4, 0.7),
                'twi': twi * 0.7,
                'dep': dep - 1
            })
            
            child.extend(L)
    
    return [[dep, [[p[0] + xof, p[1] + yof, p[2]] for p in P]]] + child
