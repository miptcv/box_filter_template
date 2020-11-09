import sys, os.path, cv2, numpy as np


def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    height = len(img)
    width = len(img[0])

    SumTable = np.ones((height, width))
    SumTable[0][0] = img[0][0]
    for j in range(1, width):
        SumTable[0][j] = SumTable[0][j - 1] + img[0][j]
    for i in range(1, height):
        SumTable[i][0] = SumTable[i - 1][0] + img[i][0]
    for i in range(1, height):
        for j in range(1, width):
            SumTable[i][j] = img[i][j] + SumTable[i - 1][j] + SumTable[i][j - 1] - SumTable[i - 1][j - 1]
    outTable = np.ones((height, width), dtype=int)
    i0 = int(); j0 = int(); i1 = int(); j1 = int()
    for i in range(0, height):
        for j in range(0, width):
            if(i - (int(h - 1) / 2) < 0):
                i0 = 0
            else:
                i0 = i - ((h - 1) // 2)
            if (i + (h / 2) > height - 1):
                i1 = height - 1
            else:
                i1 = i + (h // 2)
            if (j - ((w - 1) // 2) < 0):
                j0 = 0
            else:
                j0 = j - ((w - 1) // 2)
            if (j + (w / 2) > width - 1):
                j1 = width - 1
            else:
                j1 = j + (w // 2)
            if(j0 > 0 and i0 > 0):
                outTable[i][j] = int(float(SumTable[i1][j1] + SumTable[i0 - 1][j0 - 1] - SumTable[i0 - 1][j1] - SumTable[i1][j0 - 1]) / ((i1 - i0 + 1) * (j1 - j0 + 1)))
            elif(j0 > 0):
                outTable[i][j] = int(float(SumTable[i1][j1] - SumTable[i1][j0 - 1]) / ((i1 - i0 + 1) * (j1 - j0 + 1)))
            elif(i0 > 0):
                outTable[i][j] = int(float(SumTable[i1][j1] - SumTable[i0 - 1][j1]) / ((i1 - i0 + 1) * (j1 - j0 + 1)))
            else:
                outTable[i][j] = int(float(SumTable[i1][j1]) / ((i1 - i0 + 1) * (j1 - j0 + 1)))
    return outTable
    pass


def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])

    assert w > 0
    assert h > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    result = box_filter(img, w, h)
    cv2.imwrite(dst_path, result)

if __name__ == '__main__':
    main()