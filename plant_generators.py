import math
import random
import time
from PIL import Image, ImageDraw

from utils import (PI, v3, mapval, rand_choice, norm_rand, wtrand, rand_gaussian, sigmoid, bean, 
                  squircle, mid_pt, Noise, lerp_hue, rgba, hsv, Layer, Filter, paper, grot)
from plant_functions import leaf, stem, branch

def gen_params():
    """Generate random parameters for plants"""
    def randint(x, y):
        return math.floor(norm_rand(x, y))
    
    PAR = {}
    
    flower_shape_mask = lambda x: math.pow(math.sin(PI * x), 0.2)
    leaf_shape_mask = lambda x: math.pow(math.sin(PI * x), 0.5)
    
    PAR['flowerChance'] = rand_choice([norm_rand(0, 0.08), norm_rand(0, 0.03)])
    PAR['leafChance'] = rand_choice([0, norm_rand(0, 0.1), norm_rand(0, 0.1)])
    PAR['leafType'] = rand_choice([
        [1, randint(2, 5)],
        [2, randint(3, 7), randint(3, 8)],
        [2, randint(3, 7), randint(3, 8)]
    ])
    
    flower_shape_noise_seed = random.random() * PI
    flower_jaggedness = norm_rand(0.5, 8)
    PAR['flowerShape'] = lambda x: Noise.noise(x * flower_jaggedness, flower_shape_noise_seed) * flower_shape_mask(x)
    
    leaf_shape_noise_seed = random.random() * PI
    leaf_jaggedness = norm_rand(0.1, 40)
    leaf_pointyness = norm_rand(0.5, 1.5)
    PAR['leafShape'] = rand_choice([
        lambda x: Noise.noise(x * leaf_jaggedness, leaf_shape_noise_seed) * leaf_shape_mask(x),
        lambda x: math.pow(math.sin(PI * x), leaf_pointyness)
    ])
    
    flower_hue0 = (norm_rand(0, 180) - 130 + 360) % 360
    flower_hue1 = math.floor((flower_hue0 + norm_rand(-70, 70) + 360) % 360)
    flower_value0 = min(1, norm_rand(0.5, 1.3))
    flower_value1 = min(1, norm_rand(0.5, 1.3))
    flower_saturation0 = norm_rand(0, 1.1 - flower_value0)
    flower_saturation1 = norm_rand(0, 1.1 - flower_value1)
    
    PAR['flowerColor'] = {
        'min': [flower_hue0, flower_saturation0, flower_value0, norm_rand(0.8, 1)],
        'max': [flower_hue1, flower_saturation1, flower_value1, norm_rand(0.5, 1)]
    }
    
    PAR['leafColor'] = {
        'min': [norm_rand(10, 200), norm_rand(0.05, 0.4), norm_rand(0.3, 0.7), norm_rand(0.8, 1)],
        'max': [norm_rand(10, 200), norm_rand(0.05, 0.4), norm_rand(0.3, 0.7), norm_rand(0.8, 1)]
    }
    
    curve_coeff0 = [norm_rand(-0.5, 0.5), norm_rand(5, 10)]
    curve_coeff1 = [random.random() * PI, norm_rand(1, 5)]
    curve_coeff2 = [random.random() * PI, norm_rand(5, 15)]
    curve_coeff3 = [random.random() * PI, norm_rand(1, 5)]
    curve_coeff4 = [random.random() * 0.5, norm_rand(0.8, 1.2)]
    
    PAR['flowerOpenCurve'] = rand_choice([
        lambda x, op: (2 + op * curve_coeff2[1]) if x < 0.1 else Noise.noise(x * 10, curve_coeff2[0]),
        lambda x, op: 0 if x < curve_coeff4[0] else 10 - x * mapval(op, 0, 1, 16, 20) * curve_coeff4[1]
    ])
    
    PAR['flowerColorCurve'] = rand_choice([
        lambda x: sigmoid(x + curve_coeff0[0], curve_coeff0[1])
    ])
    
    PAR['leafLength'] = norm_rand(30, 100)
    PAR['flowerLength'] = norm_rand(5, 55)
    PAR['pedicelLength'] = norm_rand(5, 30)
    PAR['leafWidth'] = norm_rand(5, 30)
    PAR['flowerWidth'] = norm_rand(5, 30)
    PAR['stemWidth'] = norm_rand(2, 11)
    PAR['stemBend'] = norm_rand(2, 16)
    PAR['stemLength'] = norm_rand(300, 400)
    PAR['stemCount'] = rand_choice([2, 3, 4, 5])
    PAR['sheathLength'] = rand_choice([0, norm_rand(50, 100)])
    PAR['sheathWidth'] = norm_rand(5, 15)
    PAR['shootCount'] = norm_rand(1, 7)
    PAR['shootLength'] = norm_rand(50, 180)
    PAR['leafPosition'] = rand_choice([1, 2])
    PAR['flowerPetal'] = round(mapval(PAR['flowerWidth'], 5, 50, 10, 3))
    PAR['innerLength'] = min(norm_rand(0, 20), PAR['flowerLength'] * 0.8)
    PAR['innerWidth'] = min(rand_choice([0, norm_rand(1, 8)]), PAR['flowerWidth'] * 0.8)
    PAR['innerShape'] = lambda x: math.pow(math.sin(PI * x), 1)
    
    inner_hue = norm_rand(0, 60)
    PAR['innerColor'] = {
        'min': [inner_hue, norm_rand(0.1, 0.7), norm_rand(0.5, 0.9), norm_rand(0.8, 1)],
        'max': [inner_hue, norm_rand(0.1, 0.7), norm_rand(0.5, 0.9), norm_rand(0.5, 1)]
    }
    
    PAR['branchWidth'] = norm_rand(0.4, 1.3)
    PAR['branchTwist'] = round(norm_rand(2, 5))
    PAR['branchDepth'] = rand_choice([3, 4])
    PAR['branchFork'] = rand_choice([4, 5, 6, 7])
    
    branch_hue = norm_rand(30, 60)
    branch_saturation = norm_rand(0.05, 0.3)
    branch_value = norm_rand(0.7, 0.9)
    PAR['branchColor'] = {
        'min': [branch_hue, branch_saturation, branch_value, 1],
        'max': [branch_hue, branch_saturation, branch_value, 1]
    }
    
    print("Generated plant parameters:", PAR)
    
    return PAR

def vizParams(PAR):
    """Visualize parameters into a summary string"""
    summary = "Plant Parameters:\n"
    
    # Add numeric parameters
    summary += "Numeric parameters:\n"
    for k, v in PAR.items():
        if isinstance(v, (int, float)):
            summary += f"  {k}: {v:.3f}\n"
    
    # Add color parameters
    summary += "\nColor parameters:\n"
    for k, v in PAR.items():
        if isinstance(v, dict) and any(c in k.lower() for c in ['color', 'colour']):
            summary += f"  {k}:\n"
            for ck, cv in v.items():
                summary += f"    {ck}: {cv}\n"
    
    # Could add more details for functions, but a simple representation is enough for a CLI tool
    summary += "\nFunction parameters: {}\n".format(
        ", ".join([k for k, v in PAR.items() if callable(v)])
    )
    
    return summary

def woody(args):
    """Generate a woody plant"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    PAR = args.get('PAR', gen_params())
    
    cwid = 1200
    lay0 = Layer.empty(cwid)
    lay1 = Layer.empty(cwid)
    
    PL = branch({
        'img': lay0,
        'xof': cwid * 0.5,
        'yof': cwid * 0.7,
        'wid': PAR['branchWidth'],
        'twi': PAR['branchTwist'],
        'dep': PAR['branchDepth'],
        'col': PAR['branchColor'],
        'frk': PAR['branchFork']
    })
    
    for i, branch_data in enumerate(PL):
        if i / len(PL) > 0.1:
            for j, point in enumerate(branch_data[1]):
                if random.random() < PAR['leafChance']:
                    leaf({
                        'img': lay0,
                        'xof': point[0],
                        'yof': point[1],
                        'len': PAR['leafLength'] * norm_rand(0.8, 1.2),
                        'vei': PAR['leafType'],
                        'col': PAR['leafColor'],
                        'rot': [norm_rand(-1, 1) * PI, norm_rand(-1, 1) * PI, norm_rand(-1, 1) * 0],
                        'wid': lambda x: PAR['leafShape'](x) * PAR['leafWidth'],
                        'ben': lambda x: [
                            mapval(Noise.noise(x * 1, i), 0, 1, -1, 1) * 5,
                            0,
                            mapval(Noise.noise(x * 1, i + PI), 0, 1, -1, 1) * 5
                        ]
                    })
                
                if random.random() < PAR['flowerChance']:
                    hr = [norm_rand(-1, 1) * PI, norm_rand(-1, 1) * PI, norm_rand(-1, 1) * 0]
                    
                    P_ = stem({
                        'img': lay0,
                        'xof': point[0],
                        'yof': point[1],
                        'rot': hr,
                        'len': PAR['pedicelLength'],
                        'col': {'min': [50, 1, 0.9, 1], 'max': [50, 1, 0.9, 1]},
                        'wid': lambda x: math.sin(x * PI) * x * 2 + 1,
                        'ben': lambda x: [0, 0, 0]
                    })
                    
                    op = random.random()
                    r = grot(P_, -1)
                    hhr = r
                    
                    for k in range(PAR['flowerPetal']):
                        leaf({
                            'img': lay1,
                            'flo': True,
                            'xof': point[0] + P_[-1][0],
                            'yof': point[1] + P_[-1][1],
                            'rot': [hhr[0], hhr[1], hhr[2] + k / PAR['flowerPetal'] * PI * 2],
                            'len': PAR['flowerLength'] * norm_rand(0.7, 1.3),
                            'wid': lambda x: PAR['flowerShape'](x) * PAR['flowerWidth'],
                            'vei': [0],
                            'col': PAR['flowerColor'],
                            'cof': PAR['flowerColorCurve'],
                            'ben': lambda x: [
                                PAR['flowerOpenCurve'](x, op),
                                0,
                                0
                            ]
                        })
                        
                        leaf({
                            'img': lay1,
                            'flo': True,
                            'xof': point[0] + P_[-1][0],
                            'yof': point[1] + P_[-1][1],
                            'rot': [hhr[0], hhr[1], hhr[2] + k / PAR['flowerPetal'] * PI * 2],
                            'len': PAR['innerLength'] * norm_rand(0.8, 1.2),
                            'wid': lambda x: math.sin(x * PI) * 4,
                            'vei': [0],
                            'col': PAR['innerColor'],
                            'cof': lambda x: x,
                            'ben': lambda x: [
                                PAR['flowerOpenCurve'](x, op),
                                0,
                                0
                            ]
                        })
    
    Layer.filter(lay0, Filter.fade)
    Layer.filter(lay0, Filter.wispy)
    Layer.filter(lay1, Filter.wispy)
    
    b1 = Layer.bound(lay0)
    b2 = Layer.bound(lay1)
    
    bd = {
        'xmin': min(b1['xmin'], b2['xmin']),
        'xmax': max(b1['xmax'], b2['xmax']),
        'ymin': min(b1['ymin'], b2['ymin']),
        'ymax': max(b1['ymax'], b2['ymax'])
    }
    
    xref = xof - (bd['xmin'] + bd['xmax']) / 2
    yref = yof - bd['ymax']
    
    Layer.blit(img, lay0, {'ble': 'multiply', 'xof': xref, 'yof': yref})
    Layer.blit(img, lay1, {'ble': 'normal', 'xof': xref, 'yof': yref})

def herbal(args):
    """Generate a herbaceous plant"""
    img = args.get('img')
    xof = args.get('xof', 0)
    yof = args.get('yof', 0)
    PAR = args.get('PAR', gen_params())
    
    cwid = 1200
    lay0 = Layer.empty(cwid)
    lay1 = Layer.empty(cwid)
    
    x0 = cwid * 0.5
    y0 = cwid * 0.7
    
    for i in range(PAR['stemCount']):
        r = [PI / 2, 0, norm_rand(-1, 1) * PI]
        P = stem({
            'img': lay0,
            'xof': x0,
            'yof': y0,
            'len': PAR['stemLength'] * norm_rand(0.7, 1.3),
            'rot': r,
            'wid': lambda x: PAR['stemWidth'] * (
                math.pow(math.sin(x * PI / 2 + PI / 2), 0.5) * Noise.noise(x * 10) * 0.5 + 0.5
            ),
            'ben': lambda x: [
                mapval(Noise.noise(x * 1, i), 0, 1, -1, 1) * x * PAR['stemBend'],
                0,
                mapval(Noise.noise(x * 1, i + PI), 0, 1, -1, 1) * x * PAR['stemBend']
            ]
        })
        
        if PAR['leafPosition'] == 2:
            for j in range(len(P)):
                if random.random() < PAR['leafChance'] * 2:
                    leaf({
                        'img': lay0,
                        'xof': x0 + P[j][0],
                        'yof': y0 + P[j][1],
                        'len': 2 * PAR['leafLength'] * norm_rand(0.8, 1.2),
                        'vei': PAR['leafType'],
                        'col': PAR['leafColor'],
                        'rot': [norm_rand(-1, 1) * PI, norm_rand(-1, 1) * PI, norm_rand(-1, 1) * 0],
                        'wid': lambda x: 2 * PAR['leafShape'](x) * PAR['leafWidth'],
                        'ben': lambda x: [
                            mapval(Noise.noise(x * 1, i), 0, 1, -1, 1) * 5,
                            0,
                            mapval(Noise.noise(x * 1, i + PI), 0, 1, -1, 1) * 5
                        ]
                    })
        
        hr = grot(P, -1)
        if PAR['sheathLength'] != 0:
            stem({
                'img': lay0,
                'xof': x0 + P[-1][0],
                'yof': y0 + P[-1][1],
                'rot': hr,
                'len': PAR['sheathLength'],
                'col': {'min': [60, 0.3, 0.9, 1], 'max': [60, 0.3, 0.9, 1]},
                'wid': lambda x: PAR['sheathWidth'] * (math.pow(math.sin(x * PI), 2) - x * 0.5 + 0.5),
                'ben': lambda x: [0, 0, 0]
            })
        
        for j in range(max(1, int(PAR['shootCount'] * norm_rand(0.5, 1.5)))):
            P_ = stem({
                'img': lay0,
                'xof': x0 + P[-1][0],
                'yof': y0 + P[-1][1],
                'rot': hr,
                'len': PAR['shootLength'] * norm_rand(0.5, 1.5),
                'col': {'min': [70, 0.2, 0.9, 1], 'max': [70, 0.2, 0.9, 1]},
                'wid': lambda x: 2,
                'ben': lambda x: [
                    mapval(Noise.noise(x * 1, j), 0, 1, -1, 1) * x * 10,
                    0,
                    mapval(Noise.noise(x * 1, j + PI), 0, 1, -1, 1) * x * 10
                ]
            })
            
            op = random.random()
            hhr = [norm_rand(-1, 1) * PI, norm_rand(-1, 1) * PI, norm_rand(-1, 1) * PI]
            
            for k in range(PAR['flowerPetal']):
                leaf({
                    'img': lay1,
                    'flo': True,
                    'xof': x0 + P[-1][0] + P_[-1][0],
                    'yof': y0 + P[-1][1] + P_[-1][1],
                    'rot': [hhr[0], hhr[1], hhr[2] + k / PAR['flowerPetal'] * PI * 2],
                    'len': PAR['flowerLength'] * norm_rand(0.7, 1.3) * 1.5,
                    'wid': lambda x: 1.5 * PAR['flowerShape'](x) * PAR['flowerWidth'],
                    'vei': [0],
                    'col': PAR['flowerColor'],
                    'cof': PAR['flowerColorCurve'],
                    'ben': lambda x: [
                        PAR['flowerOpenCurve'](x, op),
                        0,
                        0
                    ]
                })
                
                leaf({
                    'img': lay1,
                    'flo': True,
                    'xof': x0 + P[-1][0] + P_[-1][0],
                    'yof': y0 + P[-1][1] + P_[-1][1],
                    'rot': [hhr[0], hhr[1], hhr[2] + k / PAR['flowerPetal'] * PI * 2],
                    'len': PAR['innerLength'] * norm_rand(0.8, 1.2),
                    'wid': lambda x: math.sin(x * PI) * 4,
                    'vei': [0],
                    'col': PAR['innerColor'],
                    'cof': lambda x: x,
                    'ben': lambda x: [
                        PAR['flowerOpenCurve'](x, op),
                        0,
                        0
                    ]
                })
    
    if PAR['leafPosition'] == 1:
        for i in range(int(PAR['leafChance'] * 100)):
            leaf({
                'img': lay0,
                'xof': x0,
                'yof': y0,
                'rot': [PI / 3, 0, norm_rand(-1, 1) * PI],
                'len': 4 * PAR['leafLength'] * norm_rand(0.8, 1.2),
                'wid': lambda x: 2 * PAR['leafShape'](x) * PAR['leafWidth'],
                'vei': PAR['leafType'],
                'ben': lambda x: [
                    mapval(Noise.noise(x * 1, i), 0, 1, -1, 1) * 10,
                    0,
                    mapval(Noise.noise(x * 1, i + PI), 0, 1, -1, 1) * 10
                ]
            })
    
    Layer.filter(lay0, Filter.fade)
    Layer.filter(lay0, Filter.wispy)
    Layer.filter(lay1, Filter.wispy)
    
    b1 = Layer.bound(lay0)
    b2 = Layer.bound(lay1)
    
    bd = {
        'xmin': min(b1['xmin'], b2['xmin']),
        'xmax': max(b1['xmax'], b2['xmax']),
        'ymin': min(b1['ymin'], b2['ymin']),
        'ymax': max(b1['ymax'], b2['ymax'])
    }
    
    xref = xof - (bd['xmin'] + bd['xmax']) / 2
    yref = yof - bd['ymax']
    
    Layer.blit(img, lay0, {'ble': 'multiply', 'xof': xref, 'yof': yref})
    Layer.blit(img, lay1, {'ble': 'normal', 'xof': xref, 'yof': yref})
