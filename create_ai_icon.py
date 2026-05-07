#!/usr/bin/env python3
"""
Create a professional AI-themed icon for VV-GPT
Features neural network design with modern gradient styling
"""

from PIL import Image, ImageDraw, ImageFont
import math

def create_gradient_circle(draw, center, radius, inner_color, outer_color, alpha=255):
    """Create a gradient circle effect"""
    x, y = center
    for r in range(radius, 0, -1):
        # Calculate blend ratio
        ratio = (radius - r) / radius
        
        # Blend colors
        r_color = int(outer_color[0] + (inner_color[0] - outer_color[0]) * ratio)
        g_color = int(outer_color[1] + (inner_color[1] - outer_color[1]) * ratio)
        b_color = int(outer_color[2] + (inner_color[2] - outer_color[2]) * ratio)
        
        # Calculate alpha for smooth edges
        edge_alpha = min(alpha, int(alpha * (1 - ratio * 0.3)))
        
        color = (r_color, g_color, b_color, edge_alpha)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

def create_homemade_gpt_icon():
    """Create the main VV-GPT icon"""
    size = 256
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    
    center_x, center_y = size // 2, size // 2
    
    # Color scheme: Modern AI colors
    primary_purple = (99, 102, 241)    # Indigo
    secondary_cyan = (34, 211, 238)    # Cyan
    accent_pink = (236, 72, 153)       # Pink
    dark_bg = (30, 41, 59)             # Dark slate
    light_accent = (248, 250, 252)     # Almost white
    
    # Background circle with gradient
    create_gradient_circle(draw, (center_x, center_y), 120, 
                          primary_purple, dark_bg, alpha=200)
    
    # Neural network nodes positions
    nodes = []
    # Input layer (left)
    for i in range(4):
        y = center_y - 60 + i * 40
        nodes.append((center_x - 80, y))
    
    # Hidden layer (center) 
    for i in range(5):
        y = center_y - 80 + i * 40
        nodes.append((center_x, y))
    
    # Output layer (right)
    for i in range(3):
        y = center_y - 40 + i * 40
        nodes.append((center_x + 80, y))
    
    # Draw connections (synapses)
    connection_color = (*light_accent, 120)
    for i in range(4):  # Input to hidden
        for j in range(4, 9):
            start = nodes[i]
            end = nodes[j]
            draw.line([start, end], fill=connection_color, width=2)
    
    for i in range(4, 9):  # Hidden to output
        for j in range(9, 12):
            start = nodes[i]
            end = nodes[j]
            draw.line([start, end], fill=connection_color, width=2)
    
    # Draw nodes with different colors for each layer
    node_colors = [
        (secondary_cyan, light_accent),   # Input layer
        (primary_purple, light_accent),   # Hidden layer  
        (accent_pink, light_accent)       # Output layer
    ]
    
    layer_sizes = [4, 5, 3]
    node_idx = 0
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        for i in range(layer_size):
            node_pos = nodes[node_idx]
            outer_color, inner_color = node_colors[layer_idx]
            
            # Outer glow
            create_gradient_circle(draw, node_pos, 15, 
                                 outer_color, (outer_color[0]//3, outer_color[1]//3, outer_color[2]//3), 
                                 alpha=100)
            
            # Main node
            create_gradient_circle(draw, node_pos, 8, inner_color, outer_color, alpha=255)
            
            node_idx += 1
    
    # Central brain/AI symbol
    brain_center = (center_x, center_y + 10)
    
    # Brain outline with gradient
    create_gradient_circle(draw, brain_center, 35, accent_pink, primary_purple, alpha=200)
    create_gradient_circle(draw, brain_center, 25, light_accent, accent_pink, alpha=255)
    
    # Add "AI" text in the center
    try:
        # Try to load a modern font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    text = "AI"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = brain_center[0] - text_width // 2
    text_y = brain_center[1] - text_height // 2
    
    # Text shadow
    draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 100), font=font)
    # Main text
    draw.text((text_x, text_y), text, fill=dark_bg, font=font)
    
    # Add some sparkle effects around the brain
    sparkle_positions = [
        (center_x - 50, center_y - 50),
        (center_x + 45, center_y - 60),
        (center_x - 60, center_y + 40),
        (center_x + 55, center_y + 35)
    ]
    
    for pos in sparkle_positions:
        # Small sparkle effect
        sparkle_size = 4
        create_gradient_circle(draw, pos, sparkle_size, light_accent, secondary_cyan, alpha=180)
    
    return icon

def save_icon_formats(icon):
    """Save icon in multiple formats and sizes"""
    
    # Save as PNG for preview
    icon.save('homemade_gpt_icon.png', format='PNG')
    print("✅ Created homemade_gpt_icon.png")
    
    # Create ICO file with multiple sizes
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icons = []
    
    for icon_size in icon_sizes:
        resized_icon = icon.resize(icon_size, Image.Resampling.LANCZOS)
        icons.append(resized_icon)
    
    # Save multi-size ICO
    icons[0].save('homemade_gpt.ico', format='ICO', 
                  sizes=[(ico.width, ico.height) for ico in icons])
    print("✅ Created homemade_gpt.ico")
    
    # Also create a smaller version for taskbar
    small_icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
    small_icon.save('homemade_gpt_small.png', format='PNG')
    print("✅ Created homemade_gpt_small.png")

def main():
    print("🎨 Creating VV-GPT AI-themed icon...")
    print("-" * 40)
    
    try:
        # Create the main icon
        icon = create_homemade_gpt_icon()
        
        # Save in various formats
        save_icon_formats(icon)
        
        print("-" * 40)
        print("🎉 Icon creation completed successfully!")
        print("📁 Files created:")
        print("   • homemade_gpt.ico (for shortcuts)")
        print("   • homemade_gpt_icon.png (preview)")
        print("   • homemade_gpt_small.png (taskbar)")
        
    except Exception as e:
        print(f"❌ Error creating icon: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
