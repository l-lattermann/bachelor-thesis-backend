import grpc
import numpy as np
import yaml

from protobuf_msgs.generated import image_interface_pb2 as pb2
from protobuf_msgs.generated import image_interface_pb2_grpc as pb2_grpc


class RcCubeGrpcClient:
    def __init__(self, config_path="/app/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        rc = cfg["rc_cube"]
        self.rc_cube_ip = rc["ip"]
        self.timeout = rc["timeout_sec"]
        self.max_message_size = rc["max_message_size_mb"] * 1024 * 1024

    def get_disparity_and_left(
        self,
        left_enabled=True,
        right_enabled=False,
        disparity_enabled=True,
        disparity_error_enabled=False,
        confidence_enabled=False,
        mesh_enabled=False,
        color=True,
        timeout=None,
    ):
        timeout = timeout or self.timeout

        channel = grpc.insecure_channel(
            self.rc_cube_ip,
            options=[("grpc.max_receive_message_length", self.max_message_size)],
        )
        stub = pb2_grpc.ImageInterfaceStub(channel)

        req = pb2.ImageSetRequest(
            left_enabled=left_enabled,
            right_enabled=right_enabled,
            disparity_enabled=disparity_enabled,
            disparity_error_enabled=disparity_error_enabled,
            confidence_enabled=confidence_enabled,
            mesh_enabled=mesh_enabled,
            color=color,
        )

        try:
            image_set = next(stub.StreamImageSets(req, timeout=timeout))

            if not image_set.HasField("left"):
                raise RuntimeError("No left image in ImageSet.")
            if not image_set.HasField("disparity"):
                raise RuntimeError("No disparity image in ImageSet.")

            left = image_set.left
            disp = image_set.disparity
            disp_img = disp.image

            if not disp_img.data:
                raise RuntimeError("Empty disparity image data.")

            enc = disp_img.encoding.lower()
            if enc == "mono16":
                dtype = np.uint16
            elif enc == "mono8":
                dtype = np.uint8
            else:
                raise RuntimeError(f"Unsupported disparity encoding: {disp_img.encoding}")

            h, w = int(disp_img.height), int(disp_img.width)
            step = int(disp_img.step)
            bpp = np.dtype(dtype).itemsize

            disp_arr = np.frombuffer(disp_img.data, dtype=dtype)
            disp_arr = disp_arr.reshape(h, step // bpp)[:, :w].copy()

            left_rgb = np.frombuffer(left.data, dtype=np.uint8).reshape(
                left.height, left.width, 3
            )

            cam = {
                "width": int(left.width),
                "height": int(left.height),
                "fx": float(left.focal_length),
                "fy": float(left.focal_length),
                "cx": float(left.principal_point_u),
                "cy": float(left.principal_point_v),
            }

            disp_params = {
                "scale": float(disp.scale),
                "offset": float(disp.offset),
                "invalid": float(disp.invalid_data_value),
                "baseline_m": float(disp.baseline),
                "delta_d": float(disp.delta_d),
                "encoding": disp_img.encoding,
            }

            return left_rgb, disp_arr, cam, disp_params

        finally:
            channel.close()
    