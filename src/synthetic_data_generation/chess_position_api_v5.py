"""
Chess FEN Parser - Auto-detect Starting Positions

Strategy:
1. Analyze current piece positions to determine which piece is on which square
2. Parse target FEN
3. Move pieces from their detected starting square to target square

Usage:
    blender chess-set.blend --background --python chess_position_api_v2.py -- --fen "r4rk1/1p1bqppp/n1p1pn2/p2pN3/2PP4/P1N3P1/1P1QPPBP/R4RK1" --view black
"""

import random
import os
import bpy
import math
import json
from mathutils import Vector
import sys
import argparse
from bpy_extras.object_utils import world_to_camera_view
# Rotate the offset vector
from mathutils import Matrix
import bpy_extras

# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
OUT_DIR = "./renders"    
SEED = None 
USE_HDRI = False
HDRI_PATHS = []  
HDRI_STRENGTH_RANGE = (0.3, 1.3)
USE_OVERHEAD_LIGHTS = True
EXPOSURE_RANGE = (2.8, 2.8)
WHITE_BALANCE_WARMTH = True 
CAMERA_POS_JITTER = 0.01  
CAMERA_ROT_JITTER_DEG = 1.0
USE_DOF = False            
FSTOP_RANGE = (2.8, 8.0)
USE_COMPOSITOR = False    
LENS_DISTORT_RANGE = (-0.03, -0.01)
GRAIN_RANGE = (0.005, 0.02)
VIGNETTE_RANGE = (0.15, 0.35)
PIECE_ROT_PROB = 0.75        
PIECE_ROT_MAX_DEG = 95.0     


def seed_everything():
    if SEED is None:
        random.seed()
    else:
        random.seed(SEED)


def set_if_exists(node, socket_name, value):
    s = node.inputs.get(socket_name)
    if s is not None:
        s.default_value = value
        return True
    return False


def set_bsdf_input(bsdf, names, value):
    for n in names:
        sock = bsdf.inputs.get(n)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def point_object_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_color_management(scene):
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = random.uniform(*EXPOSURE_RANGE)
    scene.view_settings.gamma = random.uniform(0.95, 1.05)


def clear_lights():
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def kelvin_to_rgb(k):
    t = k / 100.0
    if t <= 66:
        r = 255
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255
    r = max(0, min(255, r)) / 255.0
    g = max(0, min(255, g)) / 255.0
    b = max(0, min(255, b)) / 255.0
    return (r, g, b)


def setup_even_overhead_lighting(center, scale_factor):
    clear_lights()

    h = 1.6 * scale_factor

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, center.z + h))
    key = bpy.context.active_object
    key.data.shape = 'SQUARE'
    key.data.size = 3.5 * scale_factor
    key.data.energy = 60000
    key.data.color = kelvin_to_rgb(5200)
    point_object_at(key, center)

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y, center.z + h*1.1))
    fill = bpy.context.active_object
    fill.data.shape = 'SQUARE'
    fill.data.size = 5.0 * scale_factor
    fill.data.energy = 20000
    fill.data.color = kelvin_to_rgb(4200)
    point_object_at(fill, center)


def set_world_hdri(scene):
    if not USE_HDRI or not HDRI_PATHS:
        return
    hdri_path = random.choice(HDRI_PATHS)
    if not os.path.exists(hdri_path):
        return

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")

    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    bg.inputs["Strength"].default_value = random.uniform(*HDRI_STRENGTH_RANGE)

    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def ensure_material(obj, name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True

    if obj.data.materials:
        for i in range(len(obj.data.materials)):
            obj.data.materials[i] = mat
    else:
        obj.data.materials.append(mat)
    return mat


def make_procedural_wood_chessboard(board_obj):
    mat = ensure_material(board_obj, "Board_Wood_Procedural")
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf_light = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf_dark  = nodes.new("ShaderNodeBsdfPrincipled")
    mix_shader = nodes.new("ShaderNodeMixShader")

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)

    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 8.0

    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Y'        
    wave.inputs["Scale"].default_value = 28.0
    wave.inputs["Distortion"].default_value = 7.0
    wave.inputs["Detail"].default_value = 2.0
    wave.inputs["Detail Scale"].default_value = 1.5

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6

    mix_grain = nodes.new("ShaderNodeMixRGB")
    mix_grain.blend_type = 'MIX'
    mix_grain.inputs["Fac"].default_value = 0.35

    light_wood = nodes.new("ShaderNodeRGB")
    dark_wood  = nodes.new("ShaderNodeRGB")
    light_wood.outputs[0].default_value = (0.72, 0.66, 0.20, 1.0)
    dark_wood.outputs[0].default_value  = (0.06, 0.035, 0.02, 1.0)      

    mix_sq = nodes.new("ShaderNodeMixRGB")
    mix_sq.blend_type = 'MIX'

    light_tint = nodes.new("ShaderNodeMixRGB")
    light_tint.blend_type = 'MULTIPLY'
    light_tint.inputs["Fac"].default_value = 0.45

    dark_cc = nodes.new("ShaderNodeMixRGB")
    dark_cc.blend_type = 'MULTIPLY'
    dark_cc.inputs["Fac"].default_value = 1.0
    dark_cc.inputs[2].default_value = (0.8100, 0.8000, 0.6200, 1.0)

    dark_tint = nodes.new("ShaderNodeMixRGB")
    dark_tint.blend_type = 'MULTIPLY'
    dark_tint.inputs["Fac"].default_value = 0.55

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.20

    grain_ramp = nodes.new("ShaderNodeValToRGB")
    grain_ramp.color_ramp.elements[0].position = 0.35
    grain_ramp.color_ramp.elements[1].position = 0.65

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    links.new(wave.outputs["Color"], mix_grain.inputs[1])
    links.new(noise.outputs["Color"], mix_grain.inputs[2])
    links.new(mix_grain.outputs["Color"], grain_ramp.inputs["Fac"])
    links.new(grain_ramp.outputs["Color"], bump.inputs["Height"])

    links.new(checker.outputs["Fac"], mix_sq.inputs["Fac"])

    links.new(light_wood.outputs["Color"], light_tint.inputs[1])
    links.new(grain_ramp.outputs["Color"], light_tint.inputs[2])

    links.new(dark_wood.outputs["Color"], dark_tint.inputs[1])
    links.new(grain_ramp.outputs["Color"], dark_tint.inputs[2])

    links.new(light_tint.outputs["Color"], mix_sq.inputs[1])
    links.new(dark_tint.outputs["Color"], mix_sq.inputs[2])

    links.new(light_tint.outputs["Color"], bsdf_light.inputs["Base Color"])
    links.new(dark_tint.outputs["Color"], dark_cc.inputs[1])
    links.new(dark_cc.outputs["Color"], bsdf_dark.inputs["Base Color"])

    links.new(bump.outputs["Normal"], bsdf_light.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf_dark.inputs["Normal"])

    links.new(checker.outputs["Fac"], mix_shader.inputs["Fac"])
    links.new(bsdf_light.outputs["BSDF"], mix_shader.inputs[1])
    links.new(bsdf_dark.outputs["BSDF"],  mix_shader.inputs[2])

    links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])

    bsdf_light.inputs["Roughness"].default_value = 0.22 
    set_bsdf_input(bsdf_light, ["Specular", "Specular IOR Level"], 0.35)

    set_if_exists(bsdf_light, "Clearcoat", 0.35)
    set_if_exists(bsdf_light, "Clearcoat Roughness", 0.08)

    bsdf_dark.inputs["Roughness"].default_value = random.uniform(0.22, 0.50)
    set_bsdf_input(bsdf_dark, ["Specular", "Specular IOR Level"], random.uniform(0.15, 0.42))

    set_if_exists(bsdf_dark, "Clearcoat", random.uniform(0.12, 0.28))
    set_if_exists(bsdf_dark, "Clearcoat Roughness", random.uniform(0.05, 0.21))


def make_piece_woody(obj, is_white_piece):
    matname = "Piece_Wood_Light" if is_white_piece else "Piece_Wood_Dark"
    mat = ensure_material(obj, matname)
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 25.0
    noise.inputs["Detail"].default_value = 4.0

    ramp = nodes.new("ShaderNodeValToRGB")
    
    if is_white_piece:
        base = (0.491, 0.429, 0.275) 
        tint = (1.14, 1.07, 0.84)    
        
        r = base[0] * tint[0]
        g = base[1] * tint[1]
        b = base[2] * tint[2]

        k = 0.40                        
        r, g, b = r*k, g*k, b*k

        ramp.color_ramp.elements[0].color = (r * 0.92, g * 0.92, b * 0.92, 1.0)
        ramp.color_ramp.elements[1].color = (min(r * 1.08, 1.0), min(g * 1.08, 1.0), min(b * 1.08, 1.0), 1.0)

    else:
        ramp.color_ramp.elements[0].color = (0.012, 0.008, 0.006, 1.0)
        ramp.color_ramp.elements[1].color = (0.045, 0.025, 0.015, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = random.uniform(0.5, 0.8)
    set_bsdf_input(bsdf, ["Specular", "Specular IOR Level"], random.uniform(0.25, 0.45))

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if is_white_piece:
        bsdf.inputs["Roughness"].default_value = random.uniform(0.90, 0.98)
        set_bsdf_input(bsdf, ["Specular", "Specular IOR Level"], random.uniform(0.00, 0.04))

        cc = bsdf.inputs.get("Clearcoat")
        if cc: cc.default_value = 0.0
        ccr = bsdf.inputs.get("Clearcoat Roughness")
        if ccr: ccr.default_value = 0.0

    else:
        glossy = (random.random() < 0.7) 

        if glossy:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.25, 0.55)
            set_bsdf_input(bsdf, ["Specular", "Specular IOR Level"], random.uniform(0.20, 0.45))

            cc = bsdf.inputs.get("Clearcoat")
            if cc: cc.default_value = random.uniform(0.20, 0.50)
            ccr = bsdf.inputs.get("Clearcoat Roughness")
            if ccr: ccr.default_value = random.uniform(0.04, 0.14)
        else:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.45, 0.70)
            set_bsdf_input(bsdf, ["Specular", "Specular IOR Level"], random.uniform(0.10, 0.25))

            cc = bsdf.inputs.get("Clearcoat")
            if cc: cc.default_value = random.uniform(0.05, 0.15)
            ccr = bsdf.inputs.get("Clearcoat Roughness")
            if ccr: ccr.default_value = random.uniform(0.10, 0.25)

        anis = bsdf.inputs.get("Anisotropic")
        if anis: anis.default_value = random.uniform(0.2, 0.6)
        arot = bsdf.inputs.get("Anisotropic Rotation")
        if arot: arot.default_value = random.uniform(0.0, 1.0)


def jitter_camera(cam, scale_factor):
    j = CAMERA_POS_JITTER * scale_factor
    cam.location.x += random.uniform(-j, j)
    cam.location.y += random.uniform(-j, j)
    cam.location.z += random.uniform(-j, j)

    r = math.radians(CAMERA_ROT_JITTER_DEG)
    cam.rotation_euler.x += random.uniform(-r, r)
    cam.rotation_euler.y += random.uniform(-r, r)
    cam.rotation_euler.z += random.uniform(-r, r)


def get_board_corners_2d(scene, camera, board_info):
    """
    Returns the (x, y) pixel coordinates of the 4 board corners 
    for the current camera view.
    """
    p_min = board_info['plane_min']
    p_max = board_info['plane_max']
    z = p_max.z 
    
    # Define the 4 corners in 3D (Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left)
    corners_3d = [
        Vector((p_min.x, p_min.y, z)), # Corner 1
        Vector((p_max.x, p_min.y, z)), # Corner 2
        Vector((p_max.x, p_max.y, z)), # Corner 3
        Vector((p_min.x, p_max.y, z)), # Corner 4
    ]
    
    pixels = []
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    
    for corner in corners_3d:
        # Project 3D point to 2D normalized coordinates (0.0 to 1.0)
        co_2d = world_to_camera_view(scene, camera, corner)
        
        # Flip the Y coordinate.
        pixel_x = co_2d.x * res_x
        pixel_y = (1.0 - co_2d.y) * res_y 
        
        pixels.append([pixel_x, pixel_y])
        
    return pixels


def get_board_info():
    """Get board dimensions"""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8
    
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)
    
    scale_factor = board_size / REAL_BOARD_SIZE
    
    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'scale_factor': scale_factor,
    }


def enable_gpu():
    """
    Forces Blender to use the GPU for rendering.
    Supports OptiX (NVIDIA RTX) and CUDA.
    """
    print("\n" + "="*70)
    print("ENABLING GPU")
    print("="*70)
    
    try:
        # Access the Cycles preferences
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Refresh devices to detect hardware
        cycles_prefs.refresh_devices()
        
        # Set the global compute device type
        device_type = 'CUDA'
        for device in cycles_prefs.devices:
            if device.type == 'OPTIX':
                device_type = 'OPTIX'
                break
        
        cycles_prefs.compute_device_type = device_type
        print(f"  Compute Device Type: {device_type}")
        
        # Enable the actual devices
        enabled_count = 0
        for device in cycles_prefs.devices:
            if device.type == device_type:
                device.use = True
                print(f"  Enabled: {device.name}")
                enabled_count += 1
            else:
                device.use = False
        
        # Force the scene to use these settings
        bpy.context.scene.cycles.device = 'GPU'
        
        if enabled_count == 0:
            print("  Warning: No compatible GPU devices found!")
        
    except Exception as e:
        print(f"  Error enabling GPU: {e}")


def position_to_square(pos, board_info):
    """
    Convert 3D position to chess square (e.g., 'e2')
    """
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    
    # File (a-h) from X coordinate - scene is flipped
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    # Rank (1-8) from Y coordinate
    # Higher Y = lower rank (reversed)
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1
    
    return f"{file_letter}{rank_number}"

def detect_starting_positions(board_info):
    """
    Detect which piece is on which square currently
    Returns: {piece_name: {'square': 'e2', 'piece_type': 'P'}}
    """
    print("\n" + "="*70)
    print("DETECTING STARTING POSITIONS")
    print("="*70)
    
    pieces = {}
    
    # Get all chess piece objects
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        
        name = obj.name
        
        # Determine piece type from name
        piece_type = None
        
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'  # White pawn
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'  # Black pawn
        elif 'rook' in name.lower():
            piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower():
            piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower():
            piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower():
            piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower():
            piece_type = 'K' if 'white' in name.lower() else 'k'
        
        if piece_type:
            square = position_to_square(obj.location, board_info)
            obj["piece_type"] = piece_type
            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'start_pos': obj.location.copy()
            }
            print(f"  {name:20s} → {square:4s} ({piece_type})")
    
    print(f"\n✓ Detected {len(pieces)} pieces")
    return pieces

def parse_fen(fen):
    """Parse FEN into dict {square: piece_char}"""
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    
    position = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1
    
    return position

def apply_fen(fen, starting_pieces, board_info):
    """Apply FEN by moving pieces from detected starting positions"""
    print("\n" + "="*70)
    print("APPLYING FEN")
    print("="*70)
    print(f"FEN: {fen}\n")
    
    target_position = parse_fen(fen)
    square_size = board_info['square_size']
    
    # Build reverse mapping: start_square+piece_type -> piece_name
    available_pieces = {}
    for piece_name, info in starting_pieces.items():
        key = (info['square'], info['piece_type'])
        if key not in available_pieces:
            available_pieces[key] = []
        available_pieces[key].append(piece_name)
    
    pieces_used = set()
    
    # For each target square
    for target_square, piece_type in target_position.items():
        # Find a piece of this type that's close to this square
        # (prefer pieces that are already nearby - shortest distance)
        
        candidates = []
        for piece_name, info in starting_pieces.items():
            if info['piece_type'] == piece_type and piece_name not in pieces_used:
                # Calculate distance from this piece's starting square to target
                from_square = info['square']
                from_file = ord(from_square[0]) - ord('a')
                from_rank = int(from_square[1]) - 1
                to_file = ord(target_square[0]) - ord('a')
                to_rank = int(target_square[1]) - 1
                
                distance = abs(to_file - from_file) + abs(to_rank - from_rank)
                candidates.append((distance, piece_name, from_square))
        
        if not candidates:
            print(f"  Warning: No piece of type '{piece_type}' available for {target_square}")
            continue
        
        # Use closest piece
        candidates.sort()
        _, piece_name, from_square = candidates[0]
        
        # Move piece
        obj = bpy.data.objects.get(piece_name)
        if obj:
            # Calculate movement
            from_file = ord(from_square[0]) - ord('a')
            from_rank = int(from_square[1]) - 1
            to_file = ord(target_square[0]) - ord('a')
            to_rank = int(target_square[1]) - 1
            
            file_diff = to_file - from_file
            rank_diff = to_rank - from_rank
            
            # Move: +X for files right, -Y for ranks up
            obj.location.x -= file_diff * square_size
            obj.location.y -= rank_diff * square_size
            
            # Show piece
            obj.hide_render = False
            obj.hide_viewport = False

            # Random yaw rotation around Z
            if random.random() < PIECE_ROT_PROB:
                yaw = math.radians(random.uniform(-PIECE_ROT_MAX_DEG, PIECE_ROT_MAX_DEG))
                obj.rotation_euler.z += yaw
            
            pieces_used.add(piece_name)
            
            if from_square != target_square:
                print(f"  Moved {piece_name:20s} {from_square} → {target_square}")
            else:
                print(f"  Kept {piece_name:20s} at {target_square}")
    
    # Hide unused pieces (captured)
    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True
                print(f"  Hidden {piece_name}")
    
    print(f"\n✓ Position set ({len(pieces_used)} pieces visible)")

def render_all_views(board_info, view='black'):
    """Render views from white or black perspective"""
    print("\n" + "="*70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("="*70)
    
    center = board_info['center']
    scale_factor = board_info['scale_factor']

    seed_everything()
    setup_color_management(bpy.context.scene)

    # Make board look like wood
    plane = bpy.data.objects.get("Black & white")
    if plane:
        make_procedural_wood_chessboard(plane)

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if "piece_type" not in obj:
            continue

        pt = obj["piece_type"]
        is_white = str(pt).isupper()
        make_piece_woody(obj, is_white_piece=is_white)

    
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    
    # Clean cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Setup lighting
    scene = bpy.context.scene

    set_world_hdri(scene)
    if USE_OVERHEAD_LIGHTS:
        setup_even_overhead_lighting(center, scale_factor)
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 55
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.sample_clamp_direct = 2.0
    scene.cycles.sample_clamp_indirect = 2.0
    scene.render.filter_size = 1.2
    scene.render.dither_intensity = 0.6
    
    try:
        scene.cycles.device = 'GPU'
    except:
        pass
    
    # Camera positions
    camera_z = center.z + camera_height
    
    # Flip camera positions for white view (180 degree rotation)
    if view == 'white':
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east", False),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west", False),
        ]
        z_rotation_offset = math.radians(180)
    else:  # black view (default)
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x - horizontal_offset, center.y, camera_z), "2_west", False),
            ((center.x + horizontal_offset, center.y, camera_z), "3_east", False),
        ]
        z_rotation_offset = 0
    
    for location, name, point_at_center in views:
        print(f"\nRendering: {name}")
        
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object
        
        if point_at_center:
            direction = center - cam.location
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        else:
            cam.rotation_euler = (0, 0, 0)
        
        # Apply rotation for white/black view
        cam.rotation_euler.z += z_rotation_offset
        
        cam.data.lens = LENS

        jitter_camera(cam, scale_factor)
        
        bpy.context.scene.camera = cam

        bpy.context.view_layer.update() # Ensure matrices are updated
        corners = get_board_corners_2d(bpy.context.scene, cam, board_info)
        
        # Save coordinates to JSON
        json_path = f"{OUT_DIR}/{name}.json"
        with open(json_path, 'w') as f:
            json.dump({"corners": corners}, f)
        print(f"  ✓ Saved metadata: {name}.json")

        bpy.context.scene.render.filepath = f"{OUT_DIR}/{name}.jpg"
        bpy.ops.render.render(write_still=True)
        
        print(f"  ✓ Saved: {name}.jpg")
        
        bpy.data.objects.remove(cam, do_unlink=True)
    
    print("\n✓ Rendering complete")

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    parser.add_argument('--resolution', type=int, default=1600)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Render from white or black perspective')
    
    args = parser.parse_args(argv)
    
    global RES, SAMPLES, OUT_DIR
    RES = args.resolution
    SAMPLES = args.samples

    enable_gpu()
    seed_everything()

    plane = bpy.data.objects.get("Black & white")
    print("Board obj:", plane.name if plane else None)
    if plane and plane.data.materials:
        print("Board material slots:", [m.name if m else None for m in plane.data.materials])

    cnt = 0
    for o in bpy.data.objects:
        if o.type == 'MESH' and "piece_type" in o:
            cnt += 1
    print("Pieces with piece_type property:", cnt)

    
    # Get board info
    board_info = get_board_info()
    # Fix inverted board - rotate checkerboard 90 degrees around board center
    plane = bpy.data.objects.get("Black & white")
    if plane:
        # Get board center first (before rotating)
        frame = bpy.data.objects.get("Outer frame")
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (frame_min + frame_max) / 2
        
        # Store original position
        original_pos = plane.location.copy()
        
        # Move to center, rotate, move back
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        rotated_offset = rot_matrix @ offset
        
        plane.location = center + rotated_offset
    # Detect starting positions
    starting_pieces = detect_starting_positions(board_info)
    
    # Apply FEN
    apply_fen(args.fen, starting_pieces, board_info)
    
    # Render
    render_all_views(board_info, view=args.view)

if __name__ == "__main__":
    main()
