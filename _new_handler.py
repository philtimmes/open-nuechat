    async def _python_executor_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Execute Python code in an isolated chroot sandbox with resource limits"""
        import asyncio
        import io
        import os
        import sys
        import base64
        import shutil
        import uuid
        
        code = args.get("code", "")
        output_image = args.get("output_image", False)
        output_text = args.get("output_text", False)
        output_filename = args.get("output_filename", None)
        
        # Get chat sandbox for file I/O
        chat_id = context.get("chat_id") if context else None
        if chat_id:
            sandbox_dir = get_session_sandbox(chat_id)
        else:
            sandbox_dir = os.environ.get("ARTIFACTS_DIR", "/app/data/artifacts")
            os.makedirs(sandbox_dir, exist_ok=True)
        
        # Materialize session files into sandbox
        _materialized_files = []
        if chat_id:
            try:
                session_files = get_session_files(chat_id)
                for fname, fcontent in session_files.items():
                    file_path = os.path.join(sandbox_dir, fname)
                    file_dir = os.path.dirname(file_path)
                    if file_dir and not os.path.exists(file_dir):
                        os.makedirs(file_dir, exist_ok=True)
                    if not os.path.exists(file_path):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fcontent)
                        _materialized_files.append(fname)
            except Exception as e:
                logger.warning(f"[EXECUTE_PYTHON] Failed to materialize session files: {e}")
        
        # Paths
        pip_cache_dir = "/app/data/python/packages/cache"
        os.makedirs(pip_cache_dir, exist_ok=True)
        
        exec_id = str(uuid.uuid4())[:12]
        sandbox_base = "/app/data/sandboxes"
        os.makedirs(sandbox_base, exist_ok=True)
        jail_dir = os.path.join(sandbox_base, f"exec_{exec_id}")
        chroot_dir = os.path.join(jail_dir, "root")
        work_dir = os.path.join(chroot_dir, "work")
        result_dir = os.path.join(chroot_dir, "result")
        
        # Paths inside chroot (as seen by the jailed process)
        chroot_work = "/work"
        chroot_result_file = "/result/_output.json"
        chroot_script = "/work/_script.py"
        
        bind_mounts = []
        venv_dir = os.path.join(jail_dir, "venv")
        
        try:
            # ── Get admin-allowed packages ──
            admin_allowed_packages = set()
            try:
                db = context.get("db") if context else None
                if db:
                    from app.services.settings_service import SettingsService
                    pkg_setting = await SettingsService.get(db, "python_allowed_packages")
                    if pkg_setting:
                        admin_allowed_packages = {p.strip().lower() for p in pkg_setting.split(",") if p.strip()}
            except Exception as e:
                logger.debug(f"[EXECUTE_PYTHON] Could not load allowed packages: {e}")
            
            # ── Detect imports needing install ──
            import re as _re_imports
            import_names = set()
            for line in code.split('\n'):
                line = line.strip()
                m = _re_imports.match(r'^(?:from\s+(\S+)|import\s+(\S+))', line)
                if m:
                    mod = (m.group(1) or m.group(2)).split('.')[0]
                    import_names.add(mod)
            
            builtin_modules = {
                "math", "statistics", "datetime", "random", "json", "re", "csv",
                "collections", "itertools", "functools", "io", "os", "sys",
                "decimal", "fractions", "time", "string", "textwrap", "struct",
                "hashlib", "base64", "copy", "typing", "pathlib", "tempfile",
                "urllib", "abc", "dataclasses", "enum", "operator", "cmath",
                "pprint", "numbers", "array", "bisect", "heapq", "contextlib",
            }
            preinstalled = {
                "numpy", "np", "pandas", "pd", "matplotlib", "plt", "PIL",
                "pillow", "scipy", "sklearn", "scikit-learn",
            }
            
            needs_install = []
            for mod in import_names:
                mod_lower = mod.lower()
                if mod_lower in builtin_modules or mod_lower in preinstalled:
                    continue
                if mod_lower in admin_allowed_packages:
                    needs_install.append(mod)
            
            # ── Create venv if packages needed ──
            venv_python = sys.executable
            
            if needs_install:
                logger.info(f"[EXECUTE_PYTHON] Creating venv for packages: {needs_install}")
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "venv", "--system-site-packages", venv_dir,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await proc.wait()
                
                venv_python = os.path.join(venv_dir, "bin", "python")
                venv_pip = os.path.join(venv_dir, "bin", "pip")
                
                for pkg in needs_install:
                    if pkg.lower() not in admin_allowed_packages:
                        continue
                    logger.info(f"[EXECUTE_PYTHON] Installing {pkg}")
                    proc = await asyncio.create_subprocess_exec(
                        venv_pip, "install", "--cache-dir", pip_cache_dir, pkg,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    if proc.returncode != 0:
                        logger.warning(f"[EXECUTE_PYTHON] pip install {pkg} failed: {stderr.decode()[:500]}")
            
            # ── Build chroot jail ──
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            
            can_chroot = os.geteuid() == 0
            
            if can_chroot:
                # Bind-mount read-only system dirs
                ro_mounts = ["/usr", "/lib", "/etc/alternatives"]
                if os.path.exists("/lib64"):
                    ro_mounts.append("/lib64")
                
                for src in ro_mounts:
                    dst = os.path.join(chroot_dir, src.lstrip("/"))
                    os.makedirs(dst, exist_ok=True)
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "mount", "--bind", "-o", "ro", src, dst,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        )
                        await proc.wait()
                        if proc.returncode == 0:
                            bind_mounts.append(dst)
                        else:
                            can_chroot = False
                            break
                    except Exception:
                        can_chroot = False
                        break
            
            if can_chroot:
                # Bind-mount venv site-packages if needed
                if needs_install and os.path.exists(venv_dir):
                    dst = os.path.join(chroot_dir, "venv_lib")
                    os.makedirs(dst, exist_ok=True)
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "mount", "--bind", "-o", "ro", os.path.join(venv_dir, "lib"), dst,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        )
                        await proc.wait()
                        if proc.returncode == 0:
                            bind_mounts.append(dst)
                    except Exception:
                        pass
                
                # Minimal /dev nodes
                dev_dir = os.path.join(chroot_dir, "dev")
                os.makedirs(dev_dir, exist_ok=True)
                for dev_node in ["null", "zero", "urandom"]:
                    src_dev = f"/dev/{dev_node}"
                    dst_dev = os.path.join(dev_dir, dev_node)
                    if os.path.exists(src_dev):
                        try:
                            open(dst_dev, 'w').close()
                            proc = await asyncio.create_subprocess_exec(
                                "mount", "--bind", src_dev, dst_dev,
                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                            )
                            await proc.wait()
                            if proc.returncode == 0:
                                bind_mounts.append(dst_dev)
                        except Exception:
                            pass
                
                # /tmp inside chroot
                os.makedirs(os.path.join(chroot_dir, "tmp"), exist_ok=True)
                
                # Minimal /proc
                proc_dir = os.path.join(chroot_dir, "proc")
                os.makedirs(proc_dir, exist_ok=True)
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "mount", "-t", "proc", "proc", proc_dir,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.wait()
                    if proc.returncode == 0:
                        bind_mounts.append(proc_dir)
                except Exception:
                    pass
                
                # Copy user files into chroot work dir
                for fname in os.listdir(sandbox_dir):
                    src = os.path.join(sandbox_dir, fname)
                    dst = os.path.join(work_dir, fname)
                    if os.path.isfile(src) and not fname.startswith("_exec_"):
                        shutil.copy2(src, dst)
                
                logger.info(f"[EXECUTE_PYTHON] Chroot jail ready: {chroot_dir} ({len(bind_mounts)} mounts)")
            else:
                logger.info(f"[EXECUTE_PYTHON] Chroot unavailable, using subprocess isolation")
                # Fallback paths — no chroot, run directly
                work_dir = sandbox_dir
                chroot_work = sandbox_dir
                chroot_result_file = os.path.join(result_dir, "_output.json")
                chroot_script = os.path.join(sandbox_dir, f"_exec_{exec_id}.py")
            
            # ── Write wrapper script ──
            if can_chroot:
                script_host_path = os.path.join(work_dir, "_script.py")
                result_file_host = os.path.join(result_dir, "_output.json")
            else:
                script_host_path = chroot_script
                result_file_host = chroot_result_file
            
            wrapper = f'''
import sys, os, json, base64, io
os.chdir({repr(chroot_work)})
os.environ["MPLBACKEND"] = "Agg"
os.environ["HOME"] = "/tmp"

_captured = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _captured

_result = None
_image_b64 = None
_error = None
_tb = None

try:
    _local = {{}}
    exec({repr(code)}, {{}}, _local)
    _result = _local.get("result", None)
    
    if {repr(output_image)}:
        try:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                _image_b64 = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)
        except Exception:
            pass
        if not _image_b64:
            try:
                from PIL import Image as _PILImage
                _img = _local.get("result") or _local.get("img") or _local.get("image")
                if isinstance(_img, _PILImage.Image):
                    buf = io.BytesIO()
                    _img.save(buf, format="PNG")
                    buf.seek(0)
                    _image_b64 = base64.b64encode(buf.read()).decode("utf-8")
            except Exception:
                pass

except Exception as _e:
    import traceback
    _error = str(_e)
    _tb = traceback.format_exc()

sys.stdout = _old_stdout
_output = _captured.getvalue()

_res = {{"output": _output, "result": str(_result) if _result is not None else None, "error": _error, "traceback": _tb, "image_b64": _image_b64}}
with open({repr(chroot_result_file)}, "w") as _f:
    json.dump(_res, _f)
'''
            with open(script_host_path, 'w') as f:
                f.write(wrapper)
            
            # ── Build execution command ──
            # Resource limits: 60s CPU, 2GB virtual mem, 50MB file size, max 64 processes
            rlimits = "ulimit -t 60 -v 2097152 -f 51200 -u 64 2>/dev/null;"
            
            clean_env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "MPLBACKEND": "Agg",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            
            if can_chroot:
                # Network isolation
                unshare_prefix = ""
                try:
                    tp = await asyncio.create_subprocess_exec(
                        "unshare", "--net", "true",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await tp.wait()
                    if tp.returncode == 0:
                        unshare_prefix = "unshare --net "
                except Exception:
                    pass
                
                python_bin = "/usr/bin/python3"
                pythonpath = ""
                if needs_install:
                    try:
                        for d in os.listdir(os.path.join(venv_dir, "lib")):
                            if d.startswith("python"):
                                pythonpath = f"PYTHONPATH=/venv_lib/{d}/site-packages "
                                break
                    except Exception:
                        pass
                
                cmd = f'{rlimits} {unshare_prefix}chroot {chroot_dir} /bin/sh -c "{pythonpath}{python_bin} {chroot_script}"'
            else:
                cmd = f'{rlimits} {venv_python} {chroot_script}'
            
            # ── Execute ──
            logger.info(f"[EXECUTE_PYTHON] Executing (chroot={can_chroot}): {exec_id}")
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
                cwd=sandbox_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "error": "Execution timed out (300s)"}
            
            # ── Read result ──
            if os.path.exists(result_file_host):
                with open(result_file_host, 'r') as f:
                    exec_result = json.load(f)
            else:
                err_msg = stderr.decode()[:2000] if stderr else "Execution failed with no output"
                return {"success": False, "error": err_msg}
            
            # Sync back modified files from chroot work dir
            if can_chroot and chat_id:
                for fname in os.listdir(work_dir):
                    if fname.startswith("_"):
                        continue
                    src = os.path.join(work_dir, fname)
                    dst = os.path.join(sandbox_dir, fname)
                    if os.path.isfile(src):
                        try:
                            with open(src, 'r', encoding='utf-8') as f:
                                new_content = f.read()
                            old_content = get_session_file(chat_id, fname) if fname in (_materialized_files or []) else None
                            if old_content != new_content:
                                store_session_file(chat_id, fname, new_content)
                                shutil.copy2(src, dst)
                        except Exception:
                            pass
            elif chat_id and _materialized_files:
                for fname in _materialized_files:
                    file_path = os.path.join(sandbox_dir, fname)
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                new_content = f.read()
                            old_content = get_session_file(chat_id, fname)
                            if old_content != new_content:
                                store_session_file(chat_id, fname, new_content)
                        except Exception:
                            pass
            
            # ── Build response ──
            if exec_result.get("error"):
                return {
                    "success": False,
                    "error": exec_result["error"],
                    "traceback": exec_result.get("traceback", ""),
                    "output": exec_result.get("output", ""),
                }
            
            output = exec_result.get("output", "")
            response = {
                "success": True,
                "output": output if output else None,
                "result": exec_result.get("result"),
            }
            
            if output_image and exec_result.get("image_b64"):
                response["image_base64"] = exec_result["image_b64"]
                response["image_mime_type"] = "image/png"
                response["filename"] = output_filename or "output.png"
                response["direct_to_chat"] = True
                response["image_displayed"] = True
                response["message"] = "Chart/image was generated and displayed to the user."
            elif output_image:
                response["message"] = "output_image=True was set but no image was generated."
            
            if output_text:
                response["direct_to_chat"] = True
                response["direct_text"] = output if output else str(exec_result.get("result", ""))
                response["filename"] = output_filename or "output.txt"
                response["text_displayed"] = True
                response["message"] = "Output was displayed directly to the user."
            
            return response
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[EXECUTE_PYTHON] Execution failed: {e}")
            return {"success": False, "error": str(e), "traceback": tb}
        
        finally:
            # ── Cleanup: unmount then remove ──
            for mount_path in reversed(bind_mounts):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "umount", "-l", mount_path,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.wait()
                except Exception:
                    pass
            
            if os.path.exists(jail_dir):
                try:
                    shutil.rmtree(jail_dir, ignore_errors=True)
                except Exception:
                    pass
