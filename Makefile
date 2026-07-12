# =============================================================================
# Minimal build for the C++ face-reconstruction pipeline.
# Compiles the loaders + main into one binary and runs it.
#
#   make                                # build  -> build/face_recon
#   make run                            # build + run (defaults below)
#   make run MODE=rgbd BIWI_DIR=data/BK-1/01   # override the runtime options
#   make clean                          # remove build/
#
# Dependencies (all in the dev container): Eigen, OpenCV, HDF5, HighFive.
# HighFive is header-only and is vendored once into third_party/ (see below).
# =============================================================================

# ---- runtime options (forwarded to the program as CLI args) -----------------
# Modes: rgb | rgbd | live-cpu | live-gpu | dense | full | sparse  (Biwi only)
MODE      ?= rgb
BIWI_DIR  ?= data/BK-1/01
FRAMES    ?= 30

# ---- toolchain --------------------------------------------------------------
CXX  ?= g++
STD   = -std=c++17
OPT   = -O2
WARN  = -Wall -Wextra

# ---- Eigen + OpenCV via pkg-config ------------------------------------------
PKGS       = eigen3 opencv4
PKG_CFLAGS = $(shell pkg-config --cflags $(PKGS))
PKG_LIBS   = $(shell pkg-config --libs   $(PKGS))

# ---- HDF5: pkg-config module name differs per distro -> autodetect ----------
HDF5_PKG := $(shell pkg-config --exists hdf5-serial && echo hdf5-serial || (pkg-config --exists hdf5 && echo hdf5))
ifeq ($(strip $(HDF5_PKG)),)
  # no pkg-config module -> fall back to Ubuntu's serial layout (libhdf5-dev)
  HDF5_CFLAGS = -I/usr/include/hdf5/serial
  HDF5_LIBS   = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
else
  HDF5_CFLAGS = $(shell pkg-config --cflags $(HDF5_PKG))
  HDF5_LIBS   = $(shell pkg-config --libs   $(HDF5_PKG))
endif

# ---- HighFive (header-only, pinned, vendored once into third_party/) --------
HIGHFIVE_DIR = third_party/HighFive
HIGHFIVE_INC = -I$(HIGHFIVE_DIR)/include

# ---- assemble flags ---------------------------------------------------------
CXXFLAGS = $(STD) $(OPT) $(WARN) -Iinclude $(HIGHFIVE_INC) $(PKG_CFLAGS) $(HDF5_CFLAGS)
LDLIBS   = $(PKG_LIBS) $(HDF5_LIBS) -lceres -lglog -lpthread

# ---- sources ----------------------------------------------------------------
SRCS = src/main.cpp src/BFMLoader.cpp src/BiwiLoader.cpp src/render/Renderer.cpp src/render/ProjectionUtils.cpp src/render/Lighting.cpp src/CeresFitter.cpp src/LandmarkDetector.cpp src/FaceTracker.cpp
HDRS = $(wildcard include/*.h)
BIN  = build/face_recon

# ---- optional CUDA renderer (opt-in: `make USE_CUDA=1`) ---------------------
# Adds the GPU rasteriser (src/render/cuda_raster.cu + CudaRenderer.cpp), enables
# `--mode live-gpu` and `--mode verify-gpu`, and defines USE_CUDA for the C++.
# The default build (USE_CUDA unset) is completely unchanged — no CUDA toolkit
# required. Build+run this variant on a machine with an NVIDIA GPU + CUDA.
CUDA_OBJ =
ifeq ($(USE_CUDA),1)
  CUDA_HOME ?= /usr/local/cuda
  NVCC      ?= $(CUDA_HOME)/bin/nvcc
  CUDA_ARCH ?= sm_60                 # override for your GPU, e.g. sm_86 (Ampere)
  CUDA_OBJ   = build/cuda_raster.o build/cuda_photometric.o
  CXXFLAGS  += -DUSE_CUDA
  SRCS      += src/render/CudaRenderer.cpp
  LDLIBS    += -L$(CUDA_HOME)/lib64 -lcudart
endif

# ---- rules ------------------------------------------------------------------
.PHONY: all run clean

all: $(BIN)

# order-only prereq on HighFive: clone it once if it isn't there yet
$(BIN): $(SRCS) $(HDRS) $(CUDA_OBJ) | $(HIGHFIVE_DIR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) $(CUDA_OBJ) -o $@ $(LDLIBS)

# CUDA objects (only built when USE_CUDA=1). Pure CUDA — no Eigen/OpenCV
# includes — so nvcc needs no project/pkg-config flags. -fmad=false keeps the
# arithmetic close to the CPU's (no fused multiply-add contraction).
build/%.o: src/render/%.cu
	@mkdir -p build
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) --fmad=false -c $< -o $@

run: $(BIN)
	./$(BIN) --mode $(MODE) --biwi-dir $(BIWI_DIR) --frames $(FRAMES)

# fetch the header-only HighFive once (needs git + network; ~one-time)
$(HIGHFIVE_DIR):
	git clone --depth 1 --branch v2.9.0 https://github.com/BlueBrain/HighFive.git $@

clean:
	rm -rf build
