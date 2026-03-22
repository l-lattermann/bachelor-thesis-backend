import sys
from pathlib import Path
import requests


# Calculate f(pxl) with f(mm), aperture(mm) and image width(pxl)
F_MM = 6
H_PXL = 1280 
V_PXL = 960
APERTURE_V = 4.73
APERTURE_H = 3.55


def main():
    # Post to RC Cube
    base = "/home/ubuntu/Downloads/k4l-agent-backend/experiments/fetched_imgs/"
    url = "http://172.27.5.9/api/v2/pipelines/0/test/set"

    files = {
        "left_out1_low": open("rc_cube_utils/test_imgs/left.png", "rb"),
        "right_out1_low": open("rc_cube_utils/test_imgs/right.png", "rb"),
    }

    # F_PXL = F_MM * H_PXL / APERTURE_H *1.1#1.112
    F_PXL = 3000
    data = {
        "f": f"{F_PXL}",
        "t": "0.160",           # stereo cam distance
    }

    response = requests.post(url, files=files, data=data)

    print("Status:", response.status_code)
    print("Response:", response.text)

    # Always close the files
    for f in files.values():
        f.close()

if __name__ == "__main__":
    main()