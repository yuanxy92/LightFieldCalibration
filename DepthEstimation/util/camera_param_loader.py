import numpy as np

class CameraParam:
    K = np.eye(3)     
    R = np.zeros((3, 3))
    def __init__(self, focal, R, imgsize):
        self.K[0, 0] = focal
        self.K[1, 1] = focal
        self.K[0, 2] = imgsize[0] / 2
        self.K[1, 2] = imgsize[1] / 2
        self.R = R.reshape(3, 3)

class CameraParamLoader:
    cameras = []
    def __init__(self, name):
        self.name = name
        data = np.genfromtxt(name, dtype=np.float, delimiter = ' ')
        camera_num = data.shape[0]
        for ind in range(camera_num):
            focal = data[ind, 1]
            R = data[ind, 2:11]
            camera = CameraParam(focal, R, [1211, 1011])
            self.cameras.append(camera)

# if __name__ == "__main__":
#     filename = 'E:\\data\\giga_stereo\\1\\data\\CameraParams.txt'
#     cameras = CameraParamLoader(filename) 
