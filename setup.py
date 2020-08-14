from setuptools import setup

setup(
    name='DroneBlocks Tello Python OpenCV ArUco Markers',
    version='1.0',
    url='',
    license='MIT',
    author='Dennis Baldwin',
    author_email='db@droneblocks.io',
    description='Control Tello with Python, OpenCV, ArUco Markers',
    install_requires = [
        'opencv-python==4.3.0.36',
        'opencv-contrib-python==4.3.0.36'
    ]

)