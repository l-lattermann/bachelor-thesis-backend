import grpc
import numpy as np
from PIL import Image as PILImage # Renamed to avoid conflict with protobuf message
import os
import datetime
import sys
from protobuf_msgs.generated import image_interface_pb2
from protobuf_msgs.generated import image_interface_pb2_grpc


def ensure_new_file_name(base_name):
    """
    Checks if a filename exists and appends a number if it does.
    Mimics the C++ ensureNewFileName behavior.
    """
    name_parts = os.path.splitext(base_name)
    prefix = name_parts[0]
    suffix = name_parts[1]

    if not os.path.exists(base_name):
        return base_name

    n = 1
    while n < 100: # Limit attempts to avoid infinite loop
        new_name = f"{prefix}_{n}{suffix}"
        if not os.path.exists(new_name):
            return new_name
        n += 1
    return base_name # Fallback if 100 attempts fail, might overwrite


def store_disparity_params_txt(prefix, disparity_data):
    """
    Mimics the C++ storeParamTxt functionality for disparity parameters.
    """
    image_msg = disparity_data.image

    f = image_msg.focal_length if image_msg.HasField("focal_length") else 0.0
    t = disparity_data.baseline if disparity_data.HasField("baseline") else 0.0
    u = image_msg.principal_point_u if image_msg.HasField("principal_point_u") else 0.0
    v = image_msg.principal_point_v if image_msg.HasField("principal_point_v") else 0.0

    filename = ensure_new_file_name(f"{prefix}_param.txt")
    with open(filename, 'w') as f_out:
        f_out.write("# Created by grpc_image_client_python\n")
        f_out.write(f"camera.A=[{f:.5f} 0 {u:.5f}; 0 {f:.5f} {v:.5f}; 0 0 1]\n")
        f_out.write(f"camera.height={image_msg.height}\n")
        f_out.write(f"camera.width={image_msg.width}\n")
        f_out.write(f"rho={f*t:.5f}\n")
        f_out.write(f"t={t:.5f}\n")
        f_out.write(f"camera.exposure_time={image_msg.exposure_time:.5f}\n")
        f_out.write(f"camera.gain={image_msg.gain:.5f}\n")
        f_out.write(f"camera.noise={image_msg.noise:.5f}\n")
        f_out.write(f"camera.out1_reduction={image_msg.out1_reduction:.5f}\n")
        f_out.write(f"camera.brightness={image_msg.brightness:.5f}\n")
        f_out.write(f"disp.scale={disparity_data.scale:.5f}\n")
        f_out.write(f"disp.offset={disparity_data.offset:.5f}\n")
        f_out.write(f"disp.inv={disparity_data.invalid_data_value:.5f}\n")
    print(f"Saved disparity parameters to {filename}")


def process_and_save_image(image_msg, filename_prefix, is_disparity=False, disparity_meta=None):
    """
    Processes a raw image message from gRPC and saves it as a PNG (or TIFF for raw 16-bit).
    """
    width = image_msg.width
    height = image_msg.height
    encoding = image_msg.encoding
    image_data_bytes = image_msg.data

    if not image_data_bytes:
        print(f"Warning: No image data for {filename_prefix}. Skipping save.")
        return

    try:
        if encoding == "mono8":
            np_image = np.frombuffer(image_data_bytes, dtype=np.uint8).reshape((height, width))
            pil_image = PILImage.fromarray(np_image, mode='L') # 'L' for grayscale 8-bit
            output_filename = ensure_new_file_name(f"{filename_prefix}.png")
            pil_image.save(output_filename)
            print(f"Saved mono8 image to {output_filename}")

        elif encoding == "mono16":
            np_image = np.frombuffer(image_data_bytes, dtype=np.uint16).reshape((height, width))
            if is_disparity and disparity_meta:
                # Handle invalid values and normalize for display
                invalid_val = disparity_meta.invalid_data_value
                display_image = np_image.copy()

                valid_pixels = display_image[display_image != invalid_val]
                if valid_pixels.size > 0:
                    min_val, max_val = valid_pixels.min(), valid_pixels.max()
                    if max_val > min_val:
                        # Scale valid values to 0-254, reserving 255 for invalid_val
                        display_image = ((display_image - min_val) / (max_val - min_val) * 254).astype(np.uint8)
                        display_image[np_image == invalid_val] = 255 # White for invalid
                    else: # All valid pixels have the same value
                        display_image = np.full_like(display_image, 128, dtype=np.uint8) # Grey
                        display_image[np_image == invalid_val] = 0 # Black for invalid
                else: # All pixels are invalid or max_val was 0
                    display_image = np.full_like(display_image, 0, dtype=np.uint8) # All black

                pil_image = PILImage.fromarray(display_image, mode='L')
                output_filename = ensure_new_file_name(f"{filename_prefix}.png")
                pil_image.save(output_filename)
                print(f"Saved mono16 disparity image (normalized for display) to {output_filename}")

                # Optional: Save raw 16-bit disparity data as TIFF for analytical use
                pil_image_raw = PILImage.fromarray(np_image, mode='I;16') # 'I;16' for 16-bit grayscale
                output_filename_raw = ensure_new_file_name(f"{filename_prefix}_raw.tiff")
                pil_image_raw.save(output_filename_raw)
                print(f"Saved raw mono16 disparity image to {output_filename_raw}")

            else: # Regular mono16 image (not specifically disparity, e.g., depth)
                pil_image = PILImage.fromarray(np_image, mode='I;16') # 'I;16' for 16-bit grayscale
                output_filename = ensure_new_file_name(f"{filename_prefix}.tiff") # TIFF supports 16-bit well
                pil_image.save(output_filename)
                print(f"Saved mono16 image to {output_filename}")

        elif encoding == "rgb8":
            np_image = np.frombuffer(image_data_bytes, dtype=np.uint8).reshape((height, width, 3))
            pil_image = PILImage.fromarray(np_image, mode='RGB')
            output_filename = ensure_new_file_name(f"{filename_prefix}.png")
            pil_image.save(output_filename)
            print(f"Saved rgb8 image to {output_filename}")

        else:
            print(f"Warning: Unsupported encoding '{encoding}' for {filename_prefix}. Cannot save image.")
    except Exception as e:
        print(f"Error processing/saving image {filename_prefix} with encoding {encoding}: {e}")


def stream_disparity_images(target_address, output_dir="output_images"):
    """
    Connects to the gRPC server and streams image sets,
    saving left and disparity images along with disparity parameters.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set max message size to 300MB, matching the C++ client's setting.
    # This is important for receiving large images.
    channel = grpc.insecure_channel(target_address, options=[
        ('grpc.max_receive_message_length', 300 * 1024 * 1024)
    ])
    stub = image_interface_pb2_grpc.ImageInterfaceStub(channel)

    # Create the request to enable left and disparity images
    request = image_interface_pb2.ImageSetRequest(
        left_enabled=True,
        disparity_enabled=True,
        confidence_enabled=False,
        disparity_error_enabled=False,
        mesh_enabled=False,
        color=False, # Set to True if you want RGB for left/right images
    )

    print(f"Connecting to {target_address} and requesting image sets...")
    print(f"  left_enabled: {request.left_enabled}")
    print(f"  disparity_enabled: {request.disparity_enabled}")

    try:
        # Call the RPC and iterate through the stream of ImageSet messages
        for image_set in stub.StreamImageSets(request):
            ts_sec = image_set.timestamp.seconds
            ts_nsec = image_set.timestamp.nanos
            # Format timestamp for filename
            timestamp_dt = datetime.datetime.fromtimestamp(ts_sec + ts_nsec / 1e9)
            timestamp_str = timestamp_dt.strftime('%Y%m%d_%H%M%S_%f')[:-3] # Remove last 3 digits of microseconds for brevity

            print(f"Received ImageSet timestamp: {ts_sec}.{ts_nsec:09d}")

            base_filename_prefix = os.path.join(output_dir, f"image_{timestamp_str}")

            if image_set.HasField("left"):
                process_and_save_image(image_set.left, f"{base_filename_prefix}_left")
            else:
                print("No left image in this ImageSet.")

            if image_set.HasField("disparity"):
                print(f"  Disparity image present. Encoding: {image_set.disparity.image.encoding}, "
                      f"W: {image_set.disparity.image.width}, H: {image_set.disparity.image.height}")
                
                # Process and save the disparity image (and raw if mono16)
                process_and_save_image(
                    image_set.disparity.image,
                    f"{base_filename_prefix}_disparity",
                    is_disparity=True, # Flag to enable disparity-specific processing (e.g., normalization for display)
                    disparity_meta=image_set.disparity # Pass metadata for invalid value handling
                )
                
                # Save disparity metadata to a text file
                store_disparity_params_txt(f"{base_filename_prefix}_disparity", image_set.disparity)
            else:
                print("No disparity image in this ImageSet.")

    except grpc.RpcError as e:
        print(f"StreamImageSets RPC failed: code {e.code().name}, details: {e.details()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("StreamImageSets finished or failed.")


if __name__ == '__main__':
    # Default target address from your C++ example
    # You MUST change this to your actual gRPC server address (e.g., your camera's IP)
    TARGET_ADDRESS = "172.27.5.9:50051"
    
    # Optional: Allow target address to be passed as a command-line argument
    if len(sys.argv) > 1:
        TARGET_ADDRESS = sys.argv[1]

    stream_disparity_images(TARGET_ADDRESS)