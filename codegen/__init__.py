# Copyright (c) 2023 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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
import os
import stat
import torchgen.gen


def _remove_path_safety(filepath):
    if os.path.islink(filepath):
        raise RuntimeError(f"Invalid path is a soft chain: {filepath}")
    if os.path.exists(filepath):
        os.remove(filepath)


def _write_if_changed_security(self, filename: str, contents: str) -> None:
    old_contents: Optional[str]
    filepath = os.path.realpath(filename)
    try:
        with open(filepath, 'r') as f:
            old_contents = f.read()
    except IOError:
        old_contents = None
    if contents != old_contents:
        _remove_path_safety(filepath)
        with os.fdopen(os.open(filepath, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
            f.write(contents)
        os.chmod(filepath, stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


def apply_codegen_patches():
    torchgen.gen.FileManager._write_if_changed = _write_if_changed_security


apply_codegen_patches()
