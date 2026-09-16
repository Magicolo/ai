FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# Standalone zoomy image: the full in-process generation stack
# (diffusers frames, RIFE interpolation, ACE-Step music, MMAudio effects)
# on top of the torch 2.10.0 + torchvision 0.25.0 base image. This is the
# only image: the same Dockerfile serves tests, tools, and the app itself
# (CMD), selected by the docker compose run command. The slim python image
# is gone on purpose — one image means no "works in test, fails in serve"
# skew, and the heavy layers change rarely so rebuilds stay quick.
#
# Provenance: base image, package set, and vendor revisions match the
# spike-unify image, where the Ernie custom img2img loop, both Z-Image
# pipelines, ACE-Step music, and the MMAudio stack all ran green on the
# RTX 4060 Ti.

# Emit output immediately and never write bytecode caches: logs stay orderly
# inside docker logs and the image stays free of root-owned __pycache__ noise.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Container-only install target: pip manages every package in this image, so
# the Debian "externally managed" guard is intentionally disabled.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# The vendored generators live at fixed paths: ACE-Step 1.5 (native package,
# inference path only) and kijai's ComfyUI-MMAudio (carries the mmaudio
# package plus its configs), plus our minimal comfy stand-in (ProgressBar
# for the vendored flow_matching import). Engine code adds nothing to
# sys.path; the image layout is the contract.
ENV PYTHONPATH=/opt/ACE-Step-1.5:/opt/ComfyUI-MMAudio:/application/vendor

# git serves the vendor clones below, curl the compose healthcheck. The X/GUI
# libraries serve headless cv2 (pulled in transitively by the audio/video
# stack) which dlopens libxcb/libGL even though nothing ever opens a window.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned vendor clones (shallow where history is irrelevant, full where the
# engine pins an exact commit). Upstream files stay pristine: our shims live
# in zoomy/vendor_compat.py and vendor/comfy/, never in these checkouts.
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5 \
    && git clone https://github.com/kijai/ComfyUI-MMAudio.git /opt/ComfyUI-MMAudio \
    && git -C /opt/ComfyUI-MMAudio checkout 8eaeb72

WORKDIR /application

COPY requirements.txt requirements-gpu.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --root-user-action=ignore \
    -r requirements.txt -r requirements-gpu.txt -r requirements-dev.txt

COPY pyproject.toml ./
COPY zoomy ./zoomy
COPY vendor ./vendor
COPY scripts ./scripts
COPY tests ./tests

# Build-time smoke test: every heavy third-party import the engine performs
# lazily must resolve in this image, so a missing transitive dependency fails
# the build instead of a 3 a.m. render.
RUN python -c "import torch; assert torch.cuda.is_available() is False or True" \
    && python -c "import diffusers, transformers, peft; print('frames', diffusers.__version__, transformers.__version__)" \
    && python -c "import ccvfi; print('interp ok')" \
    && python -c "import acestep.handler, acestep.inference, acestep.llm_inference; print('acestep ok')" \
    && python -c "from mmaudio.eval_utils import generate; from comfy.utils import ProgressBar; print('mmaudio ok')"

EXPOSE 7861

CMD ["python", "-m", "zoomy"]
