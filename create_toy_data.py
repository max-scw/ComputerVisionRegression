from pathlib import Path
from PIL import Image
import numpy as np

from tqdm import tqdm

if __name__ == "__main__":
    # path to export images to
    path_export = Path("datasets/toydata")
    # number of images that should be generated
    n_images = 42
    # maximum number of lines per image
    n_lines_max = 10

    # size of the images
    image_sz = (256, 256)  # px
    # thickness of the lines
    line_thickness_range = (4, 10)  # px
    # color noise of background and foreground, i.e. color of the line
    background_noise_scale = 100
    foreground_noise_scale = 50
    # ----- End user input

    # reset random seed
    np.random.seed(42)

    # create export directory if it doesn't exist
    path_export.mkdir(parents=True, exist_ok=True)

    info = []
    for i in tqdm(range(n_images)):
        # random number of lines in this image
        n_lines = np.random.randint(1, n_lines_max)

        # random line thicknesses
        th_min = min(line_thickness_range)
        th_max = max(line_thickness_range)
        line_thickness = np.random.randint(th_min, th_max, size=n_lines)
        # random location/distribution of lines on the image
        positions = np.random.randint(th_min, image_sz[0] - th_max, size=n_lines)

        # create blank image (with noise)
        image = (np.random.random(image_sz) * background_noise_scale).astype(np.uint8)
        # draw lines
        points = []
        n_lines_ = 0
        for p, th in zip(positions, line_thickness):
            x_start = round(p - th / 2)
            x_end = x_start + th

            # get all points
            pts = list(range(x_start - 1, x_end + 1))

            # ensure that the new line does not overlap with other lines
            if not any([el in points for el in pts]):
                image[x_start:x_end, :] = np.random.randint(255 - foreground_noise_scale, 255, size=(th, image_sz[1]))
                # count lines
                n_lines_ += 1

        # save image to file
        filename = path_export / f"toy_data_{i:05}_{n_lines_}.jpg"
        Image.fromarray(image).save(filename)

        info.append((filename, n_lines_))
    # write info file
    info_txt = [f"{fl.as_posix()} {n}" for fl, n in info]
    with open("info.txt", "w") as fid:
        fid.writelines("\n".join(info_txt))

