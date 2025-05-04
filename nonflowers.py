#!/usr/bin/env python3
"""
Nonflowers - Procedurally generated paintings of nonexistent flowers in Gongbi style
Based on the original JavaScript implementation by Lingdong Huang 2018
Python implementation by AI assistant 2025
"""

import os
import sys
import time
import random
import argparse
import datetime
from PIL import Image

# Import utility modules
from utils import (PI, v3, seed, Noise, paper, squircle, Layer, 
                  Filter, rand_choice, norm_rand, rand_gaussian, sigmoid, bean)
from plant_functions import leaf, stem, branch
from plant_generators import gen_params, woody, herbal, vizParams

def generate_image(args):
    """Generate a nonflower image based on the provided arguments"""
    # Set the seed if provided
    if args.seed:
        seed_value = args.seed
    else:
        seed_value = str(int(time.time() * 1000))
    print(f"Using seed: {seed_value}")
    seed(seed_value)
    
    # Create canvas
    width, height = args.width, args.height
    CTX = Layer.empty(width)
    
    # Generate plant parameters
    PAR = gen_params()
    
    # Apply parameter overrides from command line arguments
    for key, value in vars(args).items():
        if key in PAR and value is not None:
            # Skip if it's not a parameter we want to override
            if key in ['seed', 'width', 'height', 'output', 'type']:
                continue
            PAR[key] = value
            print(f"Overriding parameter {key} = {value}")
    
    # Create paper texture
    PAPER_COL0 = [1, 0.99, 0.9]
    PAPER_COL1 = [0.98, 0.91, 0.74]
    
    # Apply paper texture
    bg_paper = paper({'col': PAPER_COL1})
    
    # Tile the paper texture across the canvas
    for i in range(0, width, 512):
        for j in range(0, height, 512):
            CTX.paste(bg_paper, (i, j))
    
    # Generate plant based on the type
    if args.type == 'random':
        plant_type = random.choice(['woody', 'herbal'])
    else:
        plant_type = args.type
    
    print(f"Generating {plant_type} plant...")
    
    # Generate the plant
    if plant_type == 'woody':
        woody({
            'img': CTX,
            'xof': width // 2,
            'yof': height - 50,
            'PAR': PAR
        })
    else:  # herbal
        herbal({
            'img': CTX,
            'xof': width // 2,
            'yof': height - 50,
            'PAR': PAR
        })
    
    # Apply border
    Layer.border(CTX, squircle(0.98, 3))
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = args.output
        if not filename.endswith('.png'):
            filename += '.png'
    else:
        if not os.path.exists('output'):
            os.makedirs('output')
        filename = f"output/nonflower_{timestamp}_{seed_value}.png"
    
    # Save the image
    print(f"Saving image to {filename}")
    CTX.save(filename)
    
    # Print parameter summary
    print("\nPlant parameter summary:\n" + vizParams(PAR))
    
    print(f"\nImage generation complete! Saved to: {filename}")
    
    return CTX

def main():
    """Main entry point for the nonflowers application"""
    parser = argparse.ArgumentParser(description="Generate procedural nonexistent flowers in Gongbi style")
    
    # Basic parameters
    parser.add_argument("--seed", type=str, help="Random seed for generation")
    parser.add_argument("--width", type=int, default=600, help="Image width (default: 600)")
    parser.add_argument("--height", type=int, default=800, help="Image height (default: 800)")
    parser.add_argument("--output", type=str, help="Output file path (default: output/nonflower_[timestamp]_[seed].png)")
    parser.add_argument("--type", type=str, default="random", choices=["woody", "herbal", "random"], 
                        help="Plant type to generate (default: random)")
    
    # Plant parameters
    parser.add_argument("--stemCount", type=int, help="Number of stems")
    parser.add_argument("--stemLength", type=float, help="Length of stems")
    parser.add_argument("--stemWidth", type=float, help="Width of stems")
    parser.add_argument("--stemBend", type=float, help="Amount of stem bending")
    parser.add_argument("--leafChance", type=float, help="Chance of leaf generation (0-1)")
    parser.add_argument("--flowerChance", type=float, help="Chance of flower generation (0-1)")
    parser.add_argument("--leafLength", type=float, help="Length of leaves")
    parser.add_argument("--leafWidth", type=float, help="Width of leaves")
    parser.add_argument("--flowerLength", type=float, help="Length of flower petals")
    parser.add_argument("--flowerWidth", type=float, help="Width of flower petals")
    parser.add_argument("--flowerPetal", type=int, help="Number of flower petals")
    parser.add_argument("--branchDepth", type=int, help="Depth of branch recursion (woody plants)")
    parser.add_argument("--branchFork", type=int, help="Number of branches that fork (woody plants)")
    parser.add_argument("--branchWidth", type=float, help="Width of branches (woody plants)")
    parser.add_argument("--sheathLength", type=float, help="Length of sheaths (herbal plants)")
    parser.add_argument("--sheathWidth", type=float, help="Width of sheaths (herbal plants)")
    parser.add_argument("--shootCount", type=float, help="Number of shoots (herbal plants)")
    parser.add_argument("--shootLength", type=float, help="Length of shoots (herbal plants)")
    parser.add_argument("--leafPosition", type=int, choices=[1, 2], help="Leaf positioning pattern (1 or 2)")
    
    args = parser.parse_args()
    
    # Generate the image
    generate_image(args)

if __name__ == "__main__":
    main()
