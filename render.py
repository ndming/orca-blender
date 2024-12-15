from argparse import ArgumentParser
from pathlib import Path

import bpy

if __name__ == "__main__":
    parser = ArgumentParser(description="Render a Blender file")
    parser.add_argument('file', required=True, type=str, help="Path to the Blender file")
    parser.add_argument('-o', '--out-dir', type=str, required=True, help="Path to the output folder")
    parser.add_argument('-l', '--low-res', action="store_true", help="Render at low resolution (400x225)")
    parser.add_argument('-s', '--samples', type=int, default=1024, help="Per-pixel sample count, default: 1024")
    parser.add_argument('-b', '--motion-blur', action="store_true", help="Render with motion blur enabled")
    parser.add_argument('-fs', '--frame-start', type=int, default=0, help="The frame number to start rendering from")
    parser.add_argument('-fe', '--frame-end', type=int, default=0, help="The frame number after which rendering will end")
    args = parser.parse_args()

    # Set up input and output paths
    blender_file  = Path(args.file)
    output_folder = Path(args.out_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load the .blend file
    bpy.ops.wm.open_mainfile(filepath=str(blender_file))

    # Necessary to render with GPU
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        device.use = True

    # Set up rendering engine and output properties
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

    # Enable motion blur if required
    if args.motion_blur:
        bpy.context.scene.render.motion_blur_enabled = True
        bpy.context.scene.cycles.motion_blur_steps = 16

    # Camera and frame resolution
    bpy.context.scene.camera = bpy.data.objects.get("Camera")
    if args.low_res:
        bpy.context.scene.render.resolution_x = 400
        bpy.context.scene.render.resolution_y = 225
    else:
        bpy.context.scene.render.resolution_x = 1600
        bpy.context.scene.render.resolution_y = 900

    # Register the roughness pass
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    rough_pass_name = "Roughness"
    rough_pass = view_layer.aovs.add()
    rough_pass.name = rough_pass_name
    rough_pass.type = 'VALUE'

    # Go through each material and set up AOV output for roughness
    for material in bpy.data.materials:
        # Ensure the material uses nodes
        if not material.use_nodes:
            continue

        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Add an AOV Output node if it doesn't already exist
        aov_node = nodes.get("AOV Output")
        if not aov_node:
            aov_node = nodes.new(type="ShaderNodeOutputAOV")
            aov_node.name = "AOV Output"
        
        # Set the AOV name in the node to match our AOV
        aov_node.aov_name = rough_pass_name

        # Find the Principled BSDF node or another shader node with a roughness input
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':  # Look for Principled BSDF node
                bsdf_node = node
                break
    
        if not bsdf_node:
            continue

        # Check if the input to roughness of BSDF is connected to something
        roughness_socket = bsdf_node.inputs['Roughness']
        if roughness_socket.is_linked:
            source_socket = roughness_socket.links[0].from_socket
            links.new(source_socket, aov_node.inputs['Value'])
        else:
            aov_node.inputs['Value'].default_value = roughness_socket.default_value

    # G-buffer configurations
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    view_layer.use_pass_vector = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_mist = True
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_glossy_direct = True
    view_layer.use_pass_glossy_indirect = True
    view_layer.use_pass_glossy_color = True
    view_layer.use_pass_emit = True
    view_layer.use_pass_environment = True

    # Render all frames
    start_frame = bpy.context.scene.frame_start if args.frame_start == 0 else args.frame_start
    end_frame   = bpy.context.scene.frame_end   if args.frame_end   == 0 else args.frame_end
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = str(output_folder/f"frame-{frame:04d}")
        bpy.ops.render.render(write_still=True)
