#!/bin/bash  

mkdir -p thirdParty && cd thirdParty
install_path=$(pwd)/install
mkdir -p ${install_path}

python_prefix=$(python -c "import sys; print(sys.prefix)")  
python_include=${python_prefix}/include/python3.9/
python_lib=${python_prefix}/lib/x86_64-linux-gnu/libpython3.9.so
python_exe=${python_prefix}/bin/python3.9
python_env=/home/dengnanxing/.local/lib/python3.9/site-packages #${python_prefix}/lib/python3.9/site-packages/
numpy_include=$(python -c "import numpy; print(numpy.get_include())")  

echo ${python_env}

# build pangolin
git clone -b v0.5 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
make install -j

# build opencv-4.2.0
cd ../../
wget https://github.com/opencv/opencv/archive/4.2.0.zip
unzip 4.2.0.zip
cd opencv-4.2.0
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_path}
make install -j6

opencv_dir=${install_path}/lib/cmake/opencv4

# build orbslam2
cd ../../
cd ORB-SLAM2-PYBIND
bash build.sh ${opencv_dir} ${install_path}
cd ../


# build pybind
# build boost

# Download and extract Boost (fixing broken zip issue)
echo "Downloading Boost 1.80.0..."
wget -O boost_1_80_0.tar.gz https://downloads.sourceforge.net/project/boost/boost/1.80.0/boost_1_80_0.tar.gz

echo "Extracting Boost..."
tar -xzf boost_1_80_0.tar.gz

cd boost_1_80_0

echo "Bootstrapping Boost..."
./bootstrap.sh --with-libraries=python --prefix=${install_path} --with-python=${python_exe}

echo "Building and installing Boost..."
./b2 install --with-python include=${python_include} --prefix=${install_path}

# build orbslam_pybind
cd ../pybind
mkdir -p build && cd build

cmake .. -DPYTHON_INCLUDE_DIRS=${python_include} \
         -DPYTHON_LIBRARIES=${python_lib} \
         -DPYTHON_EXECUTABLE=${python_exe} \
         -DBoost_INCLUDE_DIRS=${install_path}/include/boost \
         -DBoost_LIBRARIES=${install_path}/lib/libboost_python39.so \
         -DORB_SLAM2_INCLUDE_DIR=${install_path}/include/ORB_SLAM2 \
         -DORB_SLAM2_LIBRARIES=${install_path}/lib/libORB_SLAM2.so \
         -DOpenCV_DIR=${install_path}/lib/cmake/opencv4 \
         -DPangolin_DIR=${install_path}/lib/cmake/Pangolin \
         -DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include} \
         -DCMAKE_INSTALL_PREFIX=${python_env}

make install -j

#export LD_LIBRARY_PATH=/home/dengnanxing/RTG-SLAM/thirdParty/install/lib:$LD_LIBRARY_PATH
