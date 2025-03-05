"""
Need: 
1: python version >= 3.10
2: git clone https://github.com/NVIDIA/TensorRT.git
3: cd ~/project/TensorRT/tools/experimental/trt-engine-explorer
4: python3 -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
5: sudo apt --yes install graphviz
"""

import os, json, argparse, subprocess
import tensorrt as trt
import trex.archiving as archiving
from typing import List, Dict, Tuple, Optional
from utils.parse_trtexec_log import parse_build_log, parse_profiling_log


def run_trtexec(trt_cmdline: List[str], writer):
    '''Execute trtexec'''
    success = False
    with writer:
        log_str = None
        try:
            log = subprocess.run(
                trt_cmdline,
                check=True,
                # Redirection
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True)
            success = True
            log_str = log.stdout
        except subprocess.CalledProcessError as err:
            log_str = err.output
        except FileNotFoundError as err:
            log_str = f"\nError: {err.strerror}: {err.filename}"
            print(log_str)
        writer.write(log_str)
    return success


def append_trtexec_args(trt_args: Dict, cmd_line: List[str]):
    for arg in trt_args:
        cmd_line.append(f"--{arg}")


def build_engine_cmd(
    args: Dict,
    onnx_path: str,
    engine_path: str,
    timing_cache_path: str
) -> Tuple[List[str], str]:
    onnx_path = f'{args.onnx_path}/{args.model}_float.onnx'
    graph_json_fname = f"{args.profile_path}.graph.json"
    cmd_line = ["trtexec",
        "--verbose",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    if args.quant:
        cmd_line.append(f'--calib={args.onnx_path}/{args.model}_calibration.cache')
        cmd_line.append('--int8')
        if args.mixed_precision:
            cmd_line.append("--fp16")
    if trt.__version__ < "10.0":
        # nvtxMode=verbose is the same as profilingVerbosity=detailed, but backward-compatible
        cmd_line.append("--nvtxMode=verbose")
        cmd_line.append("--buildOnly")
        cmd_line.append("--workspace=8192")
    else:
        cmd_line.append("--profilingVerbosity=detailed")

    append_trtexec_args(args.trtexec, cmd_line)

    build_log_fname = f"{args.profile_path}.build.log"
    return cmd_line, build_log_fname


def build_engine(
    args: Dict,
    timing_cache_path: str,
    tea: Optional[archiving.EngineArchive]
) -> bool:
    def generate_build_metadata(log_file: str, metadata_json_fname: str, tea: archiving.EngineArchive):
        """Parse trtexec engine build log file and write to a JSON file"""
        build_metadata = parse_build_log(log_file, tea)
        with archiving.get_writer(tea, metadata_json_fname) as writer:
            json_str = json.dumps(build_metadata, ensure_ascii=False, indent=4)
            writer.write(json_str)
            print(f"Engine building metadata: generated output file {metadata_json_fname}")

    def print_error(build_log_file: str):
        print("\nFailed to build the engine.")
        print(f"See logfile in: {build_log_file}\n")
        print("Troubleshooting:")
        print("1. Make sure that you are running this script in an environment "
              "which has trtexec built and accessible from $PATH.")
        print("2. If this is a Jupyter notebook, make sure the "
              " trtexec is in the $PATH of the Jupyter server.")

    print(f"Building the engine: {args.engine_path}")
    cmd_line, build_log_file = build_engine_cmd(args, args.onnx_path, args.engine_path, timing_cache_path)
    print(" ".join(cmd_line))
    if args.print_only:
        return True

    writer = archiving.get_writer(tea, build_log_file)
    success = run_trtexec(cmd_line, writer)
    if success:
        print("\nSuccessfully built the engine.\n")
        build_md_json_fname = f"{args.profile_path}.build.metadata.json"
        generate_build_metadata(build_log_file, build_md_json_fname, tea)
    else:
        print_error(build_log_file)
    return success


def profile_engine_cmd(
    args: Dict,
    engine_path:str,
    profile_path: str,
    timing_cache_path: str
):
    profiling_json_fname = f"{profile_path}.profile.json"
    graph_json_fname = f"{profile_path}.graph.json"
    timing_json_fname = f"{profile_path}.timing.json"
    cmd_line = ["trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        # Profiling affects the performance of your kernel!
        # Always run and time without profiling.
        "--separateProfileRun",
        "--useSpinWait",
        f"--loadEngine={engine_path}",
        f"--exportTimes={timing_json_fname}",
        f"--exportProfile={profiling_json_fname}",
        f"--exportLayerInfo={graph_json_fname}",
        f"--timingCacheFile={timing_cache_path}",
    ]
    if trt.__version__ < "10.0":
        cmd_line.append("--nvtxMode=verbose")
    else:
        cmd_line.append("--profilingVerbosity=detailed")

    append_trtexec_args(args.trtexec, cmd_line)

    profile_log_fname = f"{profile_path}.profile.log"
    return cmd_line, profile_log_fname


def profile_engine(
    args: Dict,
    profile_path: str,
    timing_cache_path:str,
    tea: archiving.EngineArchive,
) -> bool:
    def generate_profiling_metadata(log_file: str, metadata_json_fname: str, tea: archiving.EngineArchive):
        """Parse trtexec profiling session log file and write to a JSON file"""
        profiling_metadata = parse_profiling_log(log_file, tea)
        with archiving.get_writer(tea, metadata_json_fname) as writer:
            json_str = json.dumps(profiling_metadata, ensure_ascii=False, indent=4)
            writer.write(json_str)
            print(f"Profiling metadata: generated output file {metadata_json_fname}")

    print(f"Profiling the engine: {args.engine_path}")
    cmd_line, profile_log_file = profile_engine_cmd(args, args.engine_path, profile_path, timing_cache_path)
    print(" ".join(cmd_line))
    if args.print_only:
        return True

    writer = archiving.get_writer(tea, profile_log_file)

    #with GPUMonitor(args.monitor), GPUConfigurator(*get_gpu_config_args(args)):
    success = run_trtexec(cmd_line, writer)

    if success:
        print("\nSuccessfully profiled the engine.\n")
        profiling_md_json_fname = f"{profile_path}.profile.metadata.json"
        generate_profiling_metadata(profile_log_file, profiling_md_json_fname, tea)
    else:
        print("\nFailed to profile the engine.")
        print(f"See logfile in: {profile_log_file}\n")
    return success


def generate_engine_svg(args: Dict, profile_path: str) -> bool:
    if args.print_only:
        return

    graph_json_fname = f"{profile_path}.graph.json"
    profiling_json_fname = f"{profile_path}.profile.json"

    try:
        from utils.draw_engine import draw_engine
        print(f"Generating graph diagram: {graph_json_fname}")
        draw_engine(graph_json_fname, profiling_json_fname)
    except ModuleNotFoundError:
        print("Can't generate plan SVG graph because some package is not installed")


def create_artifacts_directory(path: str):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def process_engine(
    args: Dict,
    profile: bool,
    draw: bool
) -> bool:
    profile_path = args.profile_path
    timing_cache_path = f"{args.outdir}/timing.cache"
    tea_name = f"{args.engine_path}.tea"
    tea = archiving.EngineArchive(tea_name) if args.archive else None
    if tea: tea.open()

    success = build_engine(args, timing_cache_path, tea)
    if profile:
        success = profile_engine(args, profile_path, timing_cache_path, tea)
    if draw and success:
        success = generate_engine_svg(args, profile_path)
    if tea: tea.close()
    print(f"Artifcats directory: {args.outdir}")
    return success


def get_subcmds(args: Dict):
    all = (not args.profile_engine and not args.draw_engine)
    profile, draw = [True]*2 if all else [args.profile_engine, args.draw_engine]
    return profile, draw


def do_work(args):
    create_artifacts_directory(args.outdir)
    profile, draw = get_subcmds(args)
    process_engine(args, profile, draw)


def _make_parser(parser):
    # Positional arguments.
    parser.add_argument('--model', type=str, default='ResNet18', choices=['AlexNet', 'ResNet18'], help='model name')
    parser.add_argument('--onnx_path', default='/home/chenxin/project/Quant/deploy/onnx', help="input file (ONNX model file)")
    parser.add_argument('--outdir', default='/home/chenxin/project/Quant/deploy/onnx/profile', help="directory to store output artifacts")
    parser.add_argument('trtexec', nargs='*', default=None, 
        help="trtexec agruments (without a preceding --). For example: int8 shapes=input_ids:32x512,attention_mask:32x512")
    parser.add_argument('--quant', type=int, default=1, help='whether use quant engine')
    parser.add_argument('--mixed_precision', type=int, default=1, help='whether use mixed precision')

    # Optional arguments.
    parser.add_argument('--memory-clk-freq', default='max', help="Set memory clock frequency (MHz)")
    parser.add_argument('--compute-clk-freq', default='max', help="Set compute clock frequency (MHz)")
    parser.add_argument('--power-limit', default=None, type=int, help="Set power limit")
    parser.add_argument('--dev', default=0, help="GPU device ID")
    parser.add_argument('--dont-lock-clocks', action='store_true', help="Do not lock the clocks. "
             "If set, overrides --compute-clk-freq and --memory-clk-freq")
    parser.add_argument('--monitor', action='store_true', help="Monitor GPU temperature, power, clocks and utilization while profiling.")
    parser.add_argument('--print-only', action='store_true', help='print the command-line and exit')
    parser.add_argument('--build-engine', '-b', action='store_true', default=None, help='build the engine')
    parser.add_argument('--profile-engine', '-p', action='store_true', default=None, help='profile the engine')
    parser.add_argument('--draw-engine', '-d', action='store_true', default=None, help='draw the engine')
    parser.add_argument('--archive', action='store_true', help="create a TensorRT engine archive file (.tea)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args(_make_parser(parser))

    if not args.quant:
        engine_name = f'{args.model}.engine'
    else:
        engine_name = f'{args.model}_int8_fp16.engine' if args.mixed_precision else f'{args.model}_int8.engine'
    args.engine_path = f"{args.onnx_path}/{engine_name}"
    args.profile_path = f"{os.path.dirname(args.engine_path)}/profile/{engine_name}"
    do_work(args)
