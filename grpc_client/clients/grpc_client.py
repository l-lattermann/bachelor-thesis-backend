import os
import grpc
from PIL import Image as PILImage
import io
import time
import protobuf_msgs.generated.image_interface_pb2 as pb2
import protobuf_msgs.generated.image_interface_pb2_grpc as pb2_grpc

class GrpcImageClient:
    def __init__(self, rc_cube_ip="172.27.5.9:50051"):
        self.rc_cube_ip = rc_cube_ip

    def fetch_pointcloud_ply(self, out_ply="rc_cube_mesh.ply"):

        print(f"[DEBUG] Connecting to RC Cube at {self.rc_cube_ip}")
        channel = grpc.insecure_channel(self.rc_cube_ip)
        stub = pb2_grpc.ImageInterfaceStub(channel)

        req = pb2.ImageSetRequest(
            left_enabled=True,
            right_enabled=True,          # REQUIRED for disparity
            disparity_enabled=True,      # REQUIRED for mesh generation on many stacks
            disparity_error_enabled=False,
            confidence_enabled=False,
            mesh_enabled=True,
            color=True,
        )

        # Make it cheaper; some servers require this
        req.mesh_options.max_points = 100000
        req.mesh_options.binning_method = pb2.MeshOptions.AVERAGE
        req.mesh_options.watertight = False
        req.mesh_options.textured = False

        print("[DEBUG] ImageSetRequest:")
        print(f"  left_enabled            = {req.left_enabled}")
        print(f"  right_enabled           = {req.right_enabled}")
        print(f"  disparity_enabled       = {req.disparity_enabled}")
        print(f"  disparity_error_enabled = {req.disparity_error_enabled}")
        print(f"  confidence_enabled      = {req.confidence_enabled}")
        print(f"  mesh_enabled            = {req.mesh_enabled}")
        print(f"  color                   = {req.color}")

        frame_idx = 0

        try:
            print("[DEBUG] Starting StreamImageSets()")
            for image_set in stub.StreamImageSets(req, timeout=30.0):
                frame_idx += 1
                print(f"\n[DEBUG] --- Frame #{frame_idx} ---")

                fields = image_set.ListFields()
                print("[DEBUG] Present fields:")
                for f, _ in fields:
                    print(f"  - {f.name}")

                # Timestamp
                if image_set.HasField("timestamp"):
                    ts = image_set.timestamp
                    print(f"[DEBUG] Timestamp: {ts.sec}.{ts.nsec}")

                # Mesh inspection
                if image_set.HasField("mesh"):
                    mesh = image_set.mesh
                    print("[DEBUG] Mesh field present")
                    print(f"  format: '{mesh.format}'")
                    print(f"  data size: {len(mesh.data)} bytes")

                    if not mesh.data:
                        print("[DEBUG] Mesh has no data, continuing stream")
                        continue

                    fmt = (mesh.format or "").lower()
                    if fmt != "ply":
                        print(f"[ERROR] Unexpected mesh format: '{mesh.format}'")
                        continue

                    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
                    with open(out_ply, "wb") as f:
                        f.write(mesh.data)

                    print(f"[SUCCESS] Saved PLY point cloud to: {out_ply}")
                    print(f"[SUCCESS] Total frames processed: {frame_idx}")
                    return out_ply

                else:
                    print("[DEBUG] No mesh in this ImageSet")

            print("[ERROR] Stream ended without receiving mesh data")
            raise RuntimeError("Stream ended without receiving mesh data")

        except grpc.RpcError as e:
            print("[RPC ERROR]")
            print(f"  code: {e.code()}")
            print(f"  details: {e.details()}")
            raise

        finally:
            print("[DEBUG] Closing gRPC channel")
            channel.close()
        
    def fetch_left_image(self, out_path="output/left.png"):
        """Fetch one left image from RC Cube and save it as PNG."""
        target = self.rc_cube_ip
        print(f"Connecting to {target}...")



        channel = grpc.insecure_channel(target)
        stub = pb2_grpc.ImageInterfaceStub(channel)

        request = pb2.ImageSetRequest()
        request.left_enabled = True
        request.right_enabled = False
        request.disparity_enabled = False
        request.disparity_error_enabled = False
        request.confidence_enabled = False
        request.mesh_enabled = False
        request.color = True

        try:
            response_stream = stub.StreamImageSets(request)

            for image_set in response_stream:
                if image_set.HasField("left"):
                    img = image_set.left

                    # Convert raw bytes -> PIL image
                    pil_img = self._raw_bytes_to_pil(img)

                    # Ensure output dir exists
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

                    # Save
                    pil_img.save(out_path, format="PNG")
                    print(f"Saved left image to: {out_path}")

                    return out_path  # stop after first image

                print("No left image in ImageSet (continuing)...")

            raise RuntimeError("Stream ended without receiving a left image")

        except grpc.RpcError:
            print("RPC call failed:")
            raise

        finally:
            channel.close()


    def _raw_bytes_to_pil(self, img) -> PILImage.Image:
        """Convert raw RGB8 pixel data to PIL Image."""
        raw = img.data
        width = img.width
        height = img.height

        # If your encoding can vary, log it:
        # print(f"Encoding: {img.encoding}, step: {img.step}, bigendian: {img.is_bigendian}")

        if img.encoding and img.encoding.lower() != "rgb8":
            raise ValueError(f"Expected rgb8 but got {img.encoding}")

        return PILImage.frombytes("RGB", (width, height), raw)

    def test_connection(self, timeout_sec=3.0):
        """
        Tests whether the gRPC ImageInterface service is reachable and responsive.
        Does NOT depend on camera or mesh being enabled.
        """
        print(f"[TEST] Testing gRPC connection to {self.rc_cube_ip}")

        channel = grpc.insecure_channel(self.rc_cube_ip)
        stub = pb2_grpc.ImageInterfaceStub(channel)

        # Minimal request: enable left image (most basic stream)
        request = pb2.ImageSetRequest(
            left_enabled=True,
            right_enabled=False,
            disparity_enabled=False,
            disparity_error_enabled=False,
            confidence_enabled=False,
            mesh_enabled=False,
            color=False,
        )

        try:
            start = time.time()
            stream = stub.StreamImageSets(request, timeout=timeout_sec)

            print("[TEST] Waiting for first ImageSet...")
            image_set = next(stream)  # blocks until first message or timeout

            elapsed = time.time() - start
            fields = [f.name for f, _ in image_set.ListFields()]

            print("[TEST] SUCCESS")
            print(f"[TEST] Response received after {elapsed:.3f}s")
            print(f"[TEST] ImageSet fields: {fields}")

            if image_set.HasField("left"):
                print(
                    f"[TEST] Left image: "
                    f"{image_set.left.width}x{image_set.left.height}, "
                    f"encoding={image_set.left.encoding}, "
                    f"bytes={len(image_set.left.data)}"
                )

            return True

        except grpc.RpcError as e:
            print("[TEST] FAILED")
            print(f"[TEST] gRPC error code: {e.code()}")
            print(f"[TEST] gRPC error details: {e.details()}")
            return False

        except StopIteration:
            print("[TEST] FAILED: stream ended without messages")
            return False

        finally:
            channel.close()

    def test_stream_capabilities(self, out_dir="output", base_timeout=25.0):
        """
        Runs a step-by-step capability test against StreamImageSets:
        A0: connectivity + left-only (should be fast)
        A1: left+right
        A2: left+right+disparity
        A3: mesh-only (with mesh_options)
        A4: left+right+disparity+mesh (with mesh_options)

        Saves first received left/right images (raw bytes) and mesh PLY if available.
        Returns a dict with results per test.
        """
        os.makedirs(out_dir, exist_ok=True)

        def _probe(req, label, timeout):
            channel = grpc.insecure_channel(self.rc_cube_ip)
            stub = pb2_grpc.ImageInterfaceStub(channel)

            print(f"\n[{label}] Request:")
            print("TIMEOUT = ", base_timeout)
            print(f"  left={req.left_enabled} right={req.right_enabled} disp={req.disparity_enabled} mesh={req.mesh_enabled} color={req.color}")
            if req.mesh_enabled:
                mo = req.mesh_options
                print(f"  mesh_options: max_points={mo.max_points} binning_method={mo.binning_method} watertight={mo.watertight} textured={mo.textured}")

            try:
                t0 = time.time()
                stream = stub.StreamImageSets(req, timeout=timeout)
                msg = next(stream)
                dt = time.time() - t0

                fields = [f.name for f, _ in msg.ListFields()]
                print(f"[{label}] OK after {dt:.3f}s; fields={fields}")

                result = {
                    "ok": True,
                    "latency_s": dt,
                    "fields": fields,
                    "has_left": msg.HasField("left"),
                    "has_right": msg.HasField("right"),
                    "has_disparity": msg.HasField("disparity"),
                    "has_mesh": msg.HasField("mesh"),
                }

                if msg.HasField("left"):
                    left = msg.left
                    result["left"] = {
                        "w": left.width, "h": left.height, "encoding": left.encoding, "bytes": len(left.data)
                    }
                    # save raw bytes for inspection
                    left_path = os.path.join(out_dir, f"{label}_left_{left.width}x{left.height}_{left.encoding}.bin")
                    with open(left_path, "wb") as f:
                        f.write(left.data)
                    result["left_raw_path"] = left_path
                    print(f"[{label}] left: {left.width}x{left.height} {left.encoding} bytes={len(left.data)} saved={left_path}")

                if msg.HasField("right"):
                    right = msg.right
                    result["right"] = {
                        "w": right.width, "h": right.height, "encoding": right.encoding, "bytes": len(right.data)
                    }
                    right_path = os.path.join(out_dir, f"{label}_right_{right.width}x{right.height}_{right.encoding}.bin")
                    with open(right_path, "wb") as f:
                        f.write(right.data)
                    result["right_raw_path"] = right_path
                    print(f"[{label}] right: {right.width}x{right.height} {right.encoding} bytes={len(right.data)} saved={right_path}")

                if msg.HasField("disparity"):
                    disp = msg.disparity
                    result["disparity"] = {
                        "scale": disp.scale,
                        "offset": disp.offset,
                        "baseline": disp.baseline,
                        "delta_d": disp.delta_d,
                        "invalid_data_value": disp.invalid_data_value,
                        "image": {
                            "w": disp.image.width,
                            "h": disp.image.height,
                            "encoding": disp.image.encoding,
                            "bytes": len(disp.image.data),
                        },
                    }
                    disp_path = os.path.join(out_dir, f"{label}_disparity_{disp.image.width}x{disp.image.height}_{disp.image.encoding}.bin")
                    with open(disp_path, "wb") as f:
                        f.write(disp.image.data)
                    result["disparity_raw_path"] = disp_path
                    print(f"[{label}] disparity: {disp.image.width}x{disp.image.height} {disp.image.encoding} bytes={len(disp.image.data)} saved={disp_path}")

                if msg.HasField("mesh"):
                    mesh = msg.mesh
                    result["mesh"] = {"format": mesh.format, "bytes": len(mesh.data)}
                    if mesh.data and (mesh.format or "").lower() == "ply":
                        ply_path = os.path.join(out_dir, f"{label}_mesh.ply")
                        with open(ply_path, "wb") as f:
                            f.write(mesh.data)
                        result["mesh_ply_path"] = ply_path
                        print(f"[{label}] mesh: format={mesh.format} bytes={len(mesh.data)} saved={ply_path}")
                    else:
                        print(f"[{label}] mesh present but empty or non-PLY (format={mesh.format}, bytes={len(mesh.data)})")

                return result

            except grpc.RpcError as e:
                print(f"[{label}] FAIL: {e.code()} {e.details()}")
                return {"ok": False, "code": str(e.code()), "details": e.details()}
            except StopIteration:
                print(f"[{label}] FAIL: stream ended without messages")
                return {"ok": False, "code": "StopIteration", "details": "stream ended without messages"}
            finally:
                channel.close()

        results = {}

        # A0: left-only (fast sanity)
        req = pb2.ImageSetRequest(
            left_enabled=True, right_enabled=False,
            disparity_enabled=False, disparity_error_enabled=False,
            confidence_enabled=False, mesh_enabled=False,
            color=False,
        )
        results["A0_left_only"] = _probe(req, "A0_left_only", timeout=base_timeout)

        # A1: left+right
        req = pb2.ImageSetRequest(
            left_enabled=True, right_enabled=True,
            disparity_enabled=False, disparity_error_enabled=False,
            confidence_enabled=False, mesh_enabled=False,
            color=False,
        )
        results["A1_left_right"] = _probe(req, "A1_left_right", timeout=base_timeout)

        # A2: left+right+disparity
        req = pb2.ImageSetRequest(
            left_enabled=True, right_enabled=False,
            disparity_enabled=True, disparity_error_enabled=False,
            confidence_enabled=False, mesh_enabled=False,
            color=False,
        )
        results["A2_left_disparity"] = _probe(req, "A2_left_disparity", timeout=max(10.0, base_timeout))

        # A3: mesh-only (with options)
        req = pb2.ImageSetRequest(
            left_enabled=False, right_enabled=False,
            disparity_enabled=True, disparity_error_enabled=False,
            confidence_enabled=False, mesh_enabled=False,
            color=False,
        )
        req.mesh_options.max_points = 50000
        req.mesh_options.binning_method = pb2.MeshOptions.AVERAGE
        req.mesh_options.watertight = False
        req.mesh_options.textured = False
        results["A3_dispartity_only"] = _probe(req, "A3_disparity_only", timeout=max(15.0, base_timeout))

        # A4: full: left+right+disparity+mesh (with options)
        req = pb2.ImageSetRequest(
            left_enabled=True, right_enabled=True,
            disparity_enabled=True, disparity_error_enabled=False,
            confidence_enabled=False, mesh_enabled=True,
            color=False,
        )
        req.mesh_options.max_points = 50000
        req.mesh_options.binning_method = pb2.MeshOptions.AVERAGE
        req.mesh_options.watertight = False
        req.mesh_options.textured = False
        results["A4_full_mesh"] = _probe(req, "A4_full_mesh", timeout=max(20.0, base_timeout))

        print("\n=== SUMMARY ===")
        print("REQUEST TIMEOUT = ", base_timeout)
        for k, v in results.items():
            if v.get("ok"):
                print(f"{k}: OK (latency {v.get('latency_s'):.3f}s) fields={v.get('fields')}")
            else:
                print(f"{k}: FAIL ({v.get('code')}) {v.get('details')}")

        return results