# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import functools
import hashlib
from typing import (List, Dict, Optional, Set, Callable, Any, 
                    Union, TypeVar, Iterable)
import yaml

from codegen.code_template import CodeTemplate
from codegen.model import NativeFunction, assert_never
from codegen.api.types import kernel_signature
import codegen.api.cpp as cpp
from codegen.context import native_function_manager
from codegen.utils import (
    concatMap,
    context,
)

T = TypeVar('T')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha1(s.encode('latin1')).digest()
    return int.from_bytes(sha1, byteorder='little')

# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    install_dir: str
    template_dir: str
    dry_run: bool
    filenames: Set[str]

    def __init__(self, install_dir: str, template_dir: str, dry_run: bool) -> None:
        self.install_dir = install_dir
        self.template_dir = template_dir
        self.filenames = set()
        self.dry_run = dry_run

    @staticmethod
    def _write_if_changed(filename: str, contents: str) -> None:
        old_contents: Optional[str]
        try:
            with open(filename, 'r') as f:
                old_contents = f.read()
        except IOError:
            old_contents = None
        if contents != old_contents:
            with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "w") as f:
                f.write(contents)

    def write_with_template(self, filename: str, template_fn: str,
                            env_callable: Callable[[], Union[str, Dict[str, Any]]]) -> None:
        filename = '{}/{}'.format(self.install_dir, filename)
        assert filename not in self.filenames, "duplicate file write {filename}"
        self.filenames.add(filename)
        if not self.dry_run:
            env = env_callable()
            if isinstance(env, dict):
                # TODO: Update the comment reference to the correct location
                if 'generated_comment' not in env:
                    comment = "@" + "generated by tools/codegen/gen.py"
                    comment += " from {}".format(os.path.basename(template_fn))
                    env['generated_comment'] = comment
                env['legacy_th_headers'] = []
                template = _read_template(os.path.join(self.template_dir, template_fn))
                self._write_if_changed(filename, template.substitute(env))
            elif isinstance(env, str):
                self._write_if_changed(filename, env)
            else:
                assert_never(env)


    def write(self, filename: str, env_callable: Callable[[], Union[str, Union[str, Dict[str, Any]]]]) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
            self,
            filename: str,
            items: Iterable[T],
            *,
            key_fn: Callable[[T], str],
            env_callable: Callable[[T], Dict[str, List[str]]],
            num_shards: int,
            base_env: Optional[Dict[str, Any]] = None,
            sharded_keys: Set[str]
    ) -> None:

        everything: Dict[str, Any] = {'shard_id': 'Everything'}
        shards: List[Dict[str, Any]] = [{'shard_id': f'_{i}'} for i in range(num_shards)]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    assert isinstance(shard[key], list), "sharded keys in base_env must be a list"
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []


        def merge_env(into: Dict[str, List[str]], from_: Dict[str, List[str]]) -> None:
            for k, v in from_.items():
                assert k in sharded_keys, f"undeclared sharded key {k}"
                into[k] += v

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        dot_pos = filename.rfind('.')
        if dot_pos == -1:
            dot_pos = len(filename)
        base_filename = filename[:dot_pos]
        extension = filename[dot_pos:]

        for shard in all_shards:
            shard_id = shard['shard_id']
            self.write_with_template(f"{base_filename}{shard_id}{extension}",
                                     filename,
                                     lambda: shard)

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.filenames.discard(
            f"{self.install_dir}/{base_filename}Everything{extension}")

    def write_outputs(self, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script.
        """
        self._write_if_changed(
            filename,
            ''.join(name + ";" for name in sorted(self.filenames)))


SYMINT_SET = set()

def parse_native_yaml_struct(
    es: object,
) -> List[NativeFunction]:

    rs: List[NativeFunction] = []
    if not es:
        return rs

    if 'symint' not in es:
        raise AssertionError("Can't find symint in yaml.")
    if 'official' not in es:
        raise AssertionError("Can't find official in yaml.")
    if 'custom' not in es:
        raise AssertionError("Can't find custom in yaml.")

    if es['symint']:
        for e in es['symint']:
            global SYMINT_SET
            SYMINT_SET.add(e['func'].split("(")[0])

    all_funcs = []
    if es['official']:
        all_funcs += es['official']
    if es['custom']:
        all_funcs += es['custom']

    assert isinstance(all_funcs, list)

    for e in all_funcs:
        funcs = e.get("func")
        with context(lambda: f"in:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e)
            rs.append(func)

    return rs


def gen_function_declaration(
    f: NativeFunction,
) -> List[Optional[str]]:
    with native_function_manager(f):
        sig = kernel_signature(f)
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
        if f.func.is_out_fn():
            op_name += "_out"

        ret = f"{sig.decl(name=op_name)};"
    return [ret]


def gen_return(
    f: NativeFunction,
) -> List[Optional[str]]:
    with native_function_manager(f):
        sig = kernel_signature(f)
        args_exprs_str = ', '.join(a.name for a in sig.arguments())
        # print(f.func.name.name.base)
        # print(f.func.name, f.func.name.name, f.func.name.name.base)
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
        if f.func.is_out_fn():
            op_name += "_out"

        impl_name = f.impl_name
        if not f.impl_name:
            impl_name = op_name

        if "op_api" in f.impl_ns and "acl_op" in f.impl_ns:
            p = f"""{sig.defn(name=op_name)}{{
    if (at_npu::native::env::CheckJitDisable() && at_npu::native::FormatHelper::IsOpInputBaseFormat(input)) {{
        return op_api::{impl_name}({args_exprs_str});
    }} else {{
        return acl_ops::{impl_name}({args_exprs_str});
    }}
}}
"""
        elif "op_api" in f.impl_ns or "acl_op" in f.impl_ns:
            ns = f.impl_ns[0]
            p = f"""{sig.defn(name=op_name)}{{
    return {ns}::{impl_name}({args_exprs_str});
}}
"""      
        else:
            raise AssertionError(f"unknown namespace {f.impl_ns}")

    return [p]


def parse_native_yaml(
    path: str,
) -> List[Optional[str]]:

    with open(path, "r") as f:
        es = yaml.safe_load(f)

    res = parse_native_yaml_struct(es)
    backend_declarations = sorted(set(concatMap(lambda f: gen_function_declaration(f), res)))
    dispatch_registrations_body = sorted(set(concatMap(lambda f: gen_return(f), res))) 

    return backend_declarations, dispatch_registrations_body
