# ==============================
# Compiler and Flags
# ==============================
CXX      := mpic++
MPICXX   := mpic++

CPPFLAGS := -Wall -Wextra -pedantic -fopenmp -std=c++23 \
            -Iinclude -Isrc -I.

# LDLIBS   := -lfftw3 -lm -lboost_iostreams -lboost_system -ltbb

NIX_CPPFLAGS := -I$(EIGEN_INCLUDE_DIR) \
                -I$(FFTW_INCLUDE_DIR)   \
                -I$(PETSC_DIR)/include
NIX_LDFLAGS  := -L$(patsubst %/include,%/lib,$(FFTW_INCLUDE_DIR)) \
                -L$(PETSC_DIR)/lib
NIX_LDLIBS   := -lpetsc

CPPFLAGS += $(NIX_CPPFLAGS)
LDFLAGS  += $(NIX_LDFLAGS)
LDLIBS   += $(NIX_LDLIBS)

DEBUG_FLAGS := -O0 -g -U_FORTIFY_SOURCE
OPT_FLAGS   := -O3 -DNDEBUG
GEN_FLAGS   := $(OPT_FLAGS)# $(DEBUG_FLAGS) # 

# ==============================
# Directories
# ==============================
SRC_DIR   := src
INC_DIR   := include
TEST_DIR  := tests
BUILD_DIR := build

# ==============================
# Sources & Objects
# ==============================
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/src/%.o,$(SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/test/%,$(TEST_SRCS))

NP ?= 4

# ==============================
# Phony targets
# ==============================
.DEFAULT_GOAL := all
.PHONY: all tests main clean distclean format tidy

all: tests

# ==============================
# Tests
# ==============================
tests: $(TEST_BINS)
	@echo "All tests built:"
	@$(foreach t,$(TEST_BINS),echo "  $(t)";)

# Compile test object files
$(BUILD_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(GEN_FLAGS) $(CPPFLAGS) -c $< -o $@

# Link test executables
$(BUILD_DIR)/test/%: $(BUILD_DIR)/test/%.o $(OBJS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(GEN_FLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ==============================
# Library objects  (src/)
# ==============================
$(BUILD_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(GEN_FLAGS) $(CPPFLAGS) -c $< -o $@

# ==============================
# Main executable
# ==============================
MAIN_SRC  := main.cpp
MAIN_EXEC := $(BUILD_DIR)/main

main: $(MAIN_EXEC)

$(BUILD_DIR)/main.o: $(MAIN_SRC)
	@mkdir -p $(dir $@)
	$(MPICXX) $(GEN_FLAGS) $(CPPFLAGS) -c $< -o $@

$(MAIN_EXEC): $(BUILD_DIR)/main.o $(OBJS)
	@mkdir -p $(dir $@)
	$(MPICXX) $(GEN_FLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ==============================
# Run helpers
# ==============================
run_tests: tests
	@$(foreach t,$(TEST_BINS), \
		echo ">>> Running $(t)"; \
		mpirun -np $(NP) $(t); \
	)

# ==============================
# Cleaning
# ==============================
clean:
	$(RM) -r $(BUILD_DIR)

distclean: clean

# ==============================
# Code quality
# ==============================
format:
	clang-format -i include/*pp tests/*pp # src/*.cpp 
	black */*.py


TIDY       := clang-tidy
TIDY_FLAGS := --config-file=.clang-tidy

tidy:
	@echo "Running clang-tidy..."
	@rm -f log_tidy.txt
	@$(foreach src,$(SRCS) $(TEST_SRCS), \
		echo "=== Checking $(src) ===" >> log_tidy.txt; \
		$(TIDY) $(src) $(TIDY_FLAGS) -- $(CPPFLAGS) >> log_tidy.txt 2>&1; \
		echo "" >> log_tidy.txt; \
		echo "Checked $(src)"; \
	)
	@echo "clang-tidy output written to log_tidy.txt"



euler_1d: 
	mpirun -n 1 ./build/test/euler_1d -options_file conf_files/acoustic_rk.opts >>  ./build/file.log
	
plot_euler_1d: 
	python3 tools/plot.py  conf_files/acoustic_rk.opts
	 
 run_plot_euler_1d: euler_1d plot_euler_1d
