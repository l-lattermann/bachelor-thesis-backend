import os
import grpc
import numpy as np
from PIL import Image as PILImage
import datetime
import sys
import cv2

# Assume image_interface_pb2 and image_interface_pb2_grpc are generated from your .proto file
# and are accessible under the 'protobuf_msgs.generated' package.

from grpc_client.protobuf_msgs.generated import image_interface_pb2 as pb2
from grpc_client.protobuf_msgs.generated import image_interface_pb2_grpc as pb2_grpc

class RcCubeGrpcClient:
    def __init__(self, rc_cube_ip="172.27.5.9:50051", max_message_size=300 * 1024 * 1024):
        self.rc_cube_ip = rc_cube_ip
        self.max_message_size = max_message_size

    def _get_channel_and_stub(self):
        """Creates and returns a gRPC channel and stub with max message size options."""
        channel = grpc.insecure_channel(self.rc_cube_ip, options=[
            ('grpc.max_receive_message_length', self.max_message_size)
        ])
        stub = pb2_grpc.ImageInterfaceStub(channel)
        return channel, stub

    def _ensure_output_dir_exists(self, output_dir):
        """Ensures the specified output directory exists."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _ensure_new_file_name(self, base_name):
        """Appends a number to a filename if it already exists to ensure uniqueness."""
        if not os.path.exists(base_name):
            return base_name
        
        name_parts = os.path.splitext(base_name)
        prefix = name_parts[0]
        suffix = name_parts[1]
        n = 1
        while n < 100:
            new_name = f"{prefix}_{n}{suffix}"
            if not os.path.exists(new_name):
                return new_name
            n += 1
        return base_name # Fallback if too many attempts, might overwrite

    def get_point_cloud_ply(self, output_dir=None, timeout=60.0, max_points=50000,
                            binning_method=pb2.MeshOptions.AVERAGE, watertight=True, textured=True):

        self._ensure_output_dir_exists(output_dir)
        channel, stub = self._get_channel_and_stub()
        
        request = pb2.ImageSetRequest(
            left_enabled=True,
            right_enabled=False,
            disparity_enabled=False, # Not strictly required for mesh, but might influence it
            mesh_enabled=True,
            color=False, # Color is for image data, textured=True is for point cloud texture
        )

        mesh_opts = pb2.MeshOptions()
        mesh_opts.max_points = max_points
        mesh_opts.binning_method = binning_method
        mesh_opts.watertight = watertight
        mesh_opts.textured = textured
        request.mesh_options.CopyFrom(mesh_opts) # Assign the constructed MeshOptions to the request

        try:
            stream = stub.StreamImageSets(request, timeout=timeout)
            image_set = next(stream)

            if not image_set.HasField("mesh"):
                print("No mesh in ImageSet.")
                return None
            
            mesh = image_set.mesh

            # pick whichever exists in your ImageSet
            img = None
            if image_set.HasField("left"):
                img = image_set.left
            elif image_set.HasField("right"):
                img = image_set.right

            if img is None:
                raise RuntimeError("No image field present to read intrinsics.")

            cam = {
                "width": int(img.width),
                "height": int(img.height),
                "fx": float(img.focal_length),
                "fy": float(img.focal_length),
                "cx": float(img.principal_point_u),
                "cy": float(img.principal_point_v),
            }
            if not mesh.data:
                print("Mesh has no data.")
                return None
            
            if (mesh.format or "").lower() != "ply":
                print(f"Unexpected mesh format: '{mesh.format}'. Expected 'ply'.")
                return None

            if output_dir:
                timestamp_str = datetime.datetime.fromtimestamp(image_set.timestamp.sec + image_set.timestamp.nsec / 1e9).strftime('%Y%m%d_%H%M%S_%f')[:-3]
                ply_path = self._ensure_new_file_name(os.path.join(output_dir, f"point_cloud_{timestamp_str}.ply"))
                with open(ply_path, "wb") as f:
                    f.write(mesh.data)
                return mesh.data, ply_path, cam
            else:
                return mesh.data, cam 
            

        except grpc.RpcError as e:
            print(f"Error fetching point cloud: {e.code()} {e.details()}")
            return None
        except StopIteration:
            print("Stream ended without receiving any image set.")
            return None
        finally:
            channel.close()

    def get_disparity_img(self, output_dir=None, timeout=60.0):
        self._ensure_output_dir_exists(output_dir)
        channel, stub = self._get_channel_and_stub()

        request = pb2.ImageSetRequest(
            left_enabled=True,
            right_enabled=False,
            disparity_enabled=True,
            mesh_enabled=False,   # not needed for disparity
            color=True,          # mono is fine
        )

        try:
            stream = stub.StreamImageSets(request, timeout=timeout)
            image_set = next(stream)

            if not image_set.HasField("disparity"):
                raise RuntimeError("No disparity in ImageSet (disparity_enabled=True but field missing).")

            disp = image_set.disparity  # pb2.DisparityImage

            # left image (contains intrinsics)
            if not image_set.HasField("left"):
                raise RuntimeError("No left image in ImageSet (left_enabled=True but field missing).")
            left = image_set.left  # pb2.Image

            cam = {
                "width": int(left.width),
                "height": int(left.height),
                "fx": float(left.focal_length),
                "fy": float(left.focal_length),  # only one focal_length in proto
                "cx": float(left.principal_point_u),
                "cy": float(left.principal_point_v),
            }

            # disparity image is embedded as disp.image (type Image)
            disp_img = disp.image
            if disp_img is None or not disp_img.data:
                raise RuntimeError("DisparityImage.image.data is empty.")

            # Decode disparity pixels from bytes, using encoding to pick dtype
            enc = (disp_img.encoding or "").lower()
            if enc == "mono16":
                dtype = np.uint16
            elif enc == "mono8":
                dtype = np.uint8
            else:
                # If you don't know, print and fail fast
                raise RuntimeError(f"Unsupported disparity image encoding: '{disp_img.encoding}'")

            H, W = int(disp_img.height), int(disp_img.width)

            # step is bytes per row; data is step*height
            step = int(disp_img.step)
            raw = disp_img.data

            # Convert to numpy; handle potential row padding via step
            arr = np.frombuffer(raw, dtype=dtype)
            # decode left bytes into (H,W,3) uint8
            rgb = np.frombuffer(left.data, dtype=np.uint8).reshape(left.height, left.width, 3)
            bytes_per_px = np.dtype(dtype).itemsize

            expected_row_elems = step // bytes_per_px
            if expected_row_elems * H != arr.size:
                raise RuntimeError(
                    f"Size mismatch: arr.size={arr.size}, H={H}, step={step}, dtype={dtype}"
                )

            # reshape using step then crop to width
            disp_px = arr.reshape(H, expected_row_elems)[:, :W].copy()

            disp_params = {
                "scale": float(disp.scale),
                "offset": float(disp.offset),
                "invalid": float(disp.invalid_data_value),
                "baseline_m": float(disp.baseline),
                "delta_d": float(disp.delta_d),
                "encoding": disp_img.encoding,
            }

            if output_dir:
                ts = datetime.datetime.fromtimestamp(
                    image_set.timestamp.sec + image_set.timestamp.nsec / 1e9
                ).strftime("%Y%m%d_%H%M%S_%f")[:-3]
                out_path = self._ensure_new_file_name(os.path.join(output_dir, f"disparity_{ts}.bin"))
                with open(out_path, "wb") as f:
                    f.write(raw)
                out_path = self._ensure_new_file_name(
                    os.path.join(output_dir, f"disparity_{ts}.png")
                )
                disp_vis = cv2.normalize(
                    disp_px,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                cv2.imwrite(out_path, disp_vis)
                return disp_px, rgb, cam, disp_params

            return disp_px, rgb, cam, disp_params

        except grpc.RpcError as e:
            print(f"Error fetching disparity: {e.code()} {e.details()}")
            return None
        except StopIteration:
            print("Stream ended without receiving any image set.")
            return None
        finally:
            channel.close()

