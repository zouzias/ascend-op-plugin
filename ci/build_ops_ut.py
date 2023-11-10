# Copyright (c) 2023 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import subprocess
import sys
import traceback
import platform

from distutils.version import LooseVersion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def which_cmake(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    'Returns cmake command.'
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which_cmake('cmake3')
    cmake = which_cmake('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.12.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.12.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.12.0 found')


def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.abspath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def get_torch_npu_dir():
    try:
        import torch_npu
        return os.path.dirname(os.path.abspath(torch_npu.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def check_gtest_valid():
    # validation of GoogleTest path.
    gtest_path = os.path.join(BASE_DIR, '../pytorch_ut/third_party/googletest/CMakeLists.txt')
    return os.path.exists(gtest_path)


def run_cmake():
    cmake = get_cmake_command()

    if cmake is None:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: ")
    cpp_test_dir = os.path.join(BASE_DIR, "../test")
    test_build_dir = os.path.join(cpp_test_dir, "build")
    os.makedirs(test_build_dir, exist_ok=True)
    cmake_args = [
        '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir(),
        '-DTORCH_NPU_INSTALL_DIR=' + get_torch_npu_dir(),
    ]
    if check_gtest_valid():
        cmake_args.append('-DBUILD_GTEST=ON')

    build_args = ['-j', str(multiprocessing.cpu_count())]

    subprocess.check_call([cmake, cpp_test_dir] + cmake_args, cwd=test_build_dir, env=os.environ)
    subprocess.check_call(['make'] + build_args, cwd=test_build_dir, env=os.environ)


def build_op_plugin_ut():
    run_cmake()


if __name__ == "__main__":
    build_op_plugin_ut()
