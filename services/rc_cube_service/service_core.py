import grpc
import numpy as np

from protobuf_msgs.generated import image_interface_pb2 as pb2
from protobuf_msgs.generated import image_interface_pb2_grpc as pb2_grpc


class RcCubeGrpcClient:
    def __init__(self, cfg: dict = None, ip: str = None):
        if not cfg:
            raise("RC_CUBE init got no config")
        
        if not ip:
            raise("RC_CUBE init got no IP Adress")

        self.rc_cube_socket_adress = ip
        self.timeout = cfg["timeout_sec"]
        self.max_message_size = cfg["max_message_size_mb"] * 1024 * 1024

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
            self.rc_cube_socket_adress,
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
            if confidence_enabled and not image_set.HasField("confidence"):
                raise RuntimeError("No confidence image in ImageSet.")

            left_img = image_set.left
            disp = image_set.disparity
            disp_img = disp.image
            conf_img = image_set.confidence if confidence_enabled else None

            if not left_img.data:
                raise RuntimeError("Empty left image data.")
            if not disp_img.data:
                raise RuntimeError("Empty disparity image data.")
            if confidence_enabled and not conf_img.data:
                raise RuntimeError("Empty confidence image data.")

            # disparity
            disp_enc = disp_img.encoding.lower()
            if disp_enc == "mono16":
                disp_dtype = np.uint16
            elif disp_enc == "mono8":
                disp_dtype = np.uint8
            else:
                raise RuntimeError(f"Unsupported disparity encoding: {disp_img.encoding}")

            disp_h = int(disp_img.height)
            disp_w = int(disp_img.width)
            disp_step = int(disp_img.step)
            disp_bpp = np.dtype(disp_dtype).itemsize

            disp_arr = np.frombuffer(disp_img.data, dtype=disp_dtype)
            disp_arr = disp_arr.reshape(disp_h, disp_step // disp_bpp)[:, :disp_w].copy()

            # left image
            left_enc = left_img.encoding.lower()
            if color:
                if left_enc not in ("rgb8", "bgr8"):
                    raise RuntimeError(f"Unsupported left image encoding: {left_img.encoding}")

                left_h = int(left_img.height)
                left_w = int(left_img.width)
                left_step = int(left_img.step)

                left_arr = np.frombuffer(left_img.data, dtype=np.uint8)
                left_arr = left_arr.reshape(left_h, left_step)[:, : left_w * 3]
                left_rgb = left_arr.reshape(left_h, left_w, 3).copy()

                if left_enc == "bgr8":
                    left_rgb = left_rgb[:, :, ::-1]
            else:
                if left_enc != "mono8":
                    raise RuntimeError(f"Unsupported left image encoding: {left_img.encoding}")

                left_h = int(left_img.height)
                left_w = int(left_img.width)
                left_step = int(left_img.step)

                left_arr = np.frombuffer(left_img.data, dtype=np.uint8)
                left_rgb = left_arr.reshape(left_h, left_step)[:, :left_w].copy()

            # confidence image
            conf_arr = None
            if confidence_enabled:
                conf_enc = conf_img.encoding.lower()
                if conf_enc == "mono8":
                    conf_dtype = np.uint8
                elif conf_enc == "mono16":
                    conf_dtype = np.uint16
                else:
                    raise RuntimeError(f"Unsupported confidence encoding: {conf_img.encoding}")

                conf_h = int(conf_img.height)
                conf_w = int(conf_img.width)
                conf_step = int(conf_img.step)
                conf_bpp = np.dtype(conf_dtype).itemsize

                conf_arr = np.frombuffer(conf_img.data, dtype=conf_dtype)
                conf_arr = conf_arr.reshape(conf_h, conf_step // conf_bpp)[:, :conf_w].copy()

            cam = {
                "width": int(left_img.width),
                "height": int(left_img.height),
                "fx": float(left_img.focal_length),
                "fy": float(left_img.focal_length),
                "cx": float(left_img.principal_point_u),
                "cy": float(left_img.principal_point_v),
            }

            disp_params = {
                "scale": float(disp.scale),
                "offset": float(disp.offset),
                "invalid": float(disp.invalid_data_value),
                "baseline_m": float(disp.baseline),
                "delta_d": float(disp.delta_d),
                "encoding": disp_img.encoding,
            }

            return left_rgb, disp_arr, conf_arr, cam, disp_params

        finally:
            channel.close()