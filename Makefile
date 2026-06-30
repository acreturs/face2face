# =============================================================================
# Minimal build for the C++ face-reconstruction pipeline.
# Compiles the loaders + main into one binary and runs it.
#
#   make                                # build  -> build/face_recon
#   make run                            # build + run with the defaults below
#   make run NUM_IMAGES=20 RGBD=false   # override the runtime options
#   make clean                          # remove build/
#
# Dependencies (all in the dev container): Eigen, OpenCV, HDF5, HighFive.
# HighFive is header-only and is vendored once into third_party/ (see below).
# =============================================================================

# ---- runtime options (forwarded to the program as CLI args) -----------------
NUM_IMAGES ?= 1
RGBD       ?= true

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
SRCS = src/main.cpp src/BFMLoader.cpp src/PandoraLoader.cpp src/iPhoneLoader.cpp src/render/Renderer.cpp src/render/ProjectionUtils.cpp src/render/Lighting.cpp src/CeresFitter.cpp
HDRS = $(wildcard include/*.h)
BIN  = build/face_recon

# ---- rules ------------------------------------------------------------------
.PHONY: all run clean

all: $(BIN)

# order-only prereq on HighFive: clone it once if it isn't there yet
$(BIN): $(SRCS) $(HDRS) | $(HIGHFIVE_DIR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDLIBS)

run: $(BIN)
	./$(BIN)

# fetch the header-only HighFive once (needs git + network; ~one-time)
$(HIGHFIVE_DIR):
	git clone --depth 1 --branch v2.9.0 https://github.com/BlueBrain/HighFive.git $@

clean:
	rm -rf build
