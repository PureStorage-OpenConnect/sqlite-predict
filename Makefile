VERSION=$(shell cat VERSION)
DATE=$(shell date +%Y-%m-%dT%H:%M:%S%z)
SOURCE=$(shell git describe --always --dirty 2>/dev/null || echo unknown)

VERSION_MAJOR=$(shell echo $(VERSION) | cut -d. -f1)
VERSION_MINOR=$(shell echo $(VERSION) | cut -d. -f2)
VERSION_PATCH=$(shell echo $(VERSION) | cut -d. -f3 | cut -d- -f1)

# sqlite amalgamation to vendor headers from
SQLITE_YEAR=2026
SQLITE_VERSION=3530300

ifeq ($(shell uname -s),Darwin)
CONFIG_DARWIN=y
else ifeq ($(OS),Windows_NT)
CONFIG_WINDOWS=y
else
CONFIG_LINUX=y
endif

ifdef CONFIG_DARWIN
LOADABLE_EXTENSION=dylib
endif

ifdef CONFIG_LINUX
LOADABLE_EXTENSION=so
LDFLAGS+=-lm
endif

ifdef CONFIG_WINDOWS
LOADABLE_EXTENSION=dll
LDFLAGS+=-lm
endif

CC?=gcc
prefix?=dist

TARGET_LOADABLE=$(prefix)/predict0.$(LOADABLE_EXTENSION)

OBJS=sqlite-predict.c predict-forecast.c predict-receipts.c predict-tabular.c predict-distill.c vendor/sha256.c
ONNX_OBJS=$(OBJS) predict-onnx.c

# onnxruntime is a build+runtime dependency of the loadable-onnx variant
# only. Point ONNXRUNTIME_PREFIX at an install (dir with include/ + lib/),
# or install it (macOS: `brew install onnxruntime`; the prefix is auto-
# detected there). The include lives at include/ for tarball installs and
# include/onnxruntime/ for brew, so both are on the search path.
ONNXRUNTIME_PREFIX ?= $(shell brew --prefix onnxruntime 2>/dev/null)
ONNX_CFLAGS=-DSQLITE_PREDICT_ONNX -I$(ONNXRUNTIME_PREFIX)/include \
  -I$(ONNXRUNTIME_PREFIX)/include/onnxruntime
ONNX_LDFLAGS=-L$(ONNXRUNTIME_PREFIX)/lib -lonnxruntime \
  -Wl,-rpath,$(ONNXRUNTIME_PREFIX)/lib

$(prefix):
	mkdir -p $(prefix)

vendor/sqlite3ext.h:
	curl -sfo vendor/sqlite-amalgamation.zip https://sqlite.org/$(SQLITE_YEAR)/sqlite-amalgamation-$(SQLITE_VERSION).zip
	cd vendor && unzip -oj sqlite-amalgamation.zip '*/sqlite3.h' '*/sqlite3ext.h' '*/sqlite3.c' && rm sqlite-amalgamation.zip

sqlite-predict.h: sqlite-predict.h.tmpl VERSION
	sed -e 's/$${VERSION}/$(VERSION)/g' \
	    -e 's/$${DATE}/$(DATE)/g' \
	    -e 's/$${SOURCE}/$(SOURCE)/g' \
	    -e 's/$${VERSION_MAJOR}/$(VERSION_MAJOR)/g' \
	    -e 's/$${VERSION_MINOR}/$(VERSION_MINOR)/g' \
	    -e 's/$${VERSION_PATCH}/$(VERSION_PATCH)/g' \
	    $< > $@

loadable: $(prefix) vendor/sqlite3ext.h sqlite-predict.h $(OBJS)
	$(CC) -fPIC -shared -std=c99 -Wall -Wextra -Ivendor/ -I./ -O3 $(CFLAGS) $(OBJS) -o $(TARGET_LOADABLE) $(LDFLAGS)

# opt-in ONNX build: same loadable, plus predict-onnx.c linked against
# onnxruntime. Serves onnx-runtime models (distilled students, exported
# classifiers) through predict(). The core `loadable` stays zero-dependency.
loadable-onnx: $(prefix) vendor/sqlite3ext.h sqlite-predict.h $(ONNX_OBJS)
	$(CC) -fPIC -shared -std=c99 -Wall -Wextra -Ivendor/ -I./ -O3 \
	  $(ONNX_CFLAGS) $(CFLAGS) $(ONNX_OBJS) -o $(TARGET_LOADABLE) \
	  $(LDFLAGS) $(ONNX_LDFLAGS)

# GPU-enabled onnx build: adds the CUDA and TensorRT execution providers
# (fp16/int8 precision) behind -DSQLITE_PREDICT_ONNX_GPU. Running it needs
# an onnxruntime-gpu install; the provider-options symbols are in the C API
# of every onnxruntime build, so this compiles and links against the CPU
# onnxruntime for a CI compile-check. Real GPU execution is validated on the
# gated GPU job.
loadable-onnx-gpu: $(prefix) vendor/sqlite3ext.h sqlite-predict.h $(ONNX_OBJS)
	$(CC) -fPIC -shared -std=c99 -Wall -Wextra -Ivendor/ -I./ -O3 \
	  $(ONNX_CFLAGS) -DSQLITE_PREDICT_ONNX_GPU $(CFLAGS) $(ONNX_OBJS) \
	  -o $(TARGET_LOADABLE) $(LDFLAGS) $(ONNX_LDFLAGS)

debug: $(prefix) vendor/sqlite3ext.h sqlite-predict.h $(OBJS)
	$(CC) -fPIC -shared -std=c99 -Wall -Wextra -Ivendor/ -I./ -g -O0 -DSQLITE_PREDICT_DEBUG $(CFLAGS) $(OBJS) -o $(TARGET_LOADABLE) $(LDFLAGS)

test-loadable: loadable
	cd tests && uv run pytest -q

test: test-loadable

# ONNX build + its tests. The onnx-marked tests build a fixture model and
# exercise the runtime path; they self-skip when the loaded extension has
# no onnx runtime, so the core `make test` above ignores them.
test-onnx: loadable-onnx
	cd tests && uv run pytest -q

# ASan+UBSan on the C soak driver (standalone executable: no DYLD
# injection needed, macOS SIP strips it for system binaries anyway).
# Covers every operation, receipts, replay, and the error paths.
test-asan: vendor/sqlite3ext.h sqlite-predict.h
	mkdir -p $(prefix)
	clang -std=c99 -g -O1 -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -fno-sanitize-recover=undefined \
	  -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -DSQLITE_STRICT_SUBTYPE=1 \
	  -Ivendor/ -I./ \
	  tests/soak.c $(OBJS) vendor/sqlite3.c -o $(prefix)/soak-asan
	UBSAN_OPTIONS=print_stacktrace=1 ./$(prefix)/soak-asan

# ASan/UBSan soak for the onnx backend (leak-checked on Linux via LSan;
# onnxruntime's own still-reachable allocations are filtered by onnx.supp).
# Needs onnxruntime (ONNXRUNTIME_PREFIX) and the committed fixture model.
test-asan-onnx: vendor/sqlite3ext.h sqlite-predict.h
	mkdir -p $(prefix)
	clang -std=c99 -g -O1 -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -fno-sanitize-recover=undefined \
	  -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -DSQLITE_STRICT_SUBTYPE=1 \
	  $(ONNX_CFLAGS) -Ivendor/ -I./ \
	  tests/soak_onnx.c $(ONNX_OBJS) vendor/sqlite3.c \
	  -o $(prefix)/soak-onnx-asan $(ONNX_LDFLAGS)
	LSAN_OPTIONS=suppressions=tests/onnx.supp \
	  UBSAN_OPTIONS=print_stacktrace=1 \
	  ./$(prefix)/soak-onnx-asan tests/fixtures/logreg.onnx \
	  tests/fixtures/knn_incontext.onnx

# libFuzzer harness (statically links sqlite3.c; SQLITE_CORE build)
fuzz-build: vendor/sqlite3ext.h sqlite-predict.h
	mkdir -p $(prefix)
	clang -std=c99 -g -O1 -fsanitize=fuzzer,address,undefined \
	  -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -DSQLITE_STRICT_SUBTYPE=1 \
	  -Ivendor/ -I./ \
	  fuzz/fuzz_predict.c $(OBJS) vendor/sqlite3.c \
	  -o $(prefix)/fuzz_predict

fuzz: fuzz-build
	mkdir -p fuzz/corpus
	./$(prefix)/fuzz_predict -max_total_time=$${FUZZ_SECONDS:-60} \
	  -max_len=512 fuzz/corpus fuzz/seeds

# Apple clang ships no libFuzzer runtime; fuzz in a Linux container.
fuzz-docker:
	docker run --rm -v $$(pwd):/src -w /src silkeh/clang:17 bash -c "\
	  mkdir -p dist fuzz/corpus && \
	  clang -std=c99 -g -O1 -fsanitize=fuzzer,address \
	    -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -Ivendor/ -I./ \
	    fuzz/fuzz_predict.c $(OBJS) vendor/sqlite3.c \
	    -o dist/fuzz_predict_linux -lm -lpthread -ldl && \
	  ./dist/fuzz_predict_linux -max_total_time=$${FUZZ_SECONDS:-60} \
	    -max_len=512 fuzz/corpus fuzz/seeds"

# real valgrind, in a Linux container (also exercises gcc + glibc).
# Builds only the header + soak binary — no `make clean`/`make loadable`,
# which would wipe the host's dist/ through the bind mount.
test-valgrind: vendor/sqlite3ext.h sqlite-predict.h
	docker run --rm -v $$(pwd):/src -w /src gcc:13 bash -c "\
	  apt-get update -qq && apt-get install -y -qq valgrind > /dev/null && \
	  gcc -std=c99 -g -O0 -Ivendor/ -I./ -DSQLITE_CORE -DSQLITE_PREDICT_STATIC \
	    tests/soak.c $(OBJS) vendor/sqlite3.c -o dist/soak-linux -lm -lpthread -ldl && \
	  valgrind --leak-check=full --error-exitcode=9 --errors-for-leak-kinds=definite \
	    ./dist/soak-linux"

# standalone soak binary (used by CI valgrind + Windows, runnable directly)
soak: $(prefix) vendor/sqlite3ext.h sqlite-predict.h
	$(CC) -std=c99 -g -O0 -Ivendor/ -I./ -DSQLITE_CORE -DSQLITE_PREDICT_STATIC \
	  -DSQLITE_STRICT_SUBTYPE=1 \
	  tests/soak.c $(OBJS) vendor/sqlite3.c -o $(prefix)/soak $(LDFLAGS)

# WebAssembly soak (emscripten): compiles the static soak to wasm. CI runs
# it under node to prove the code is wasm-portable and correct there.
soak-wasm: $(prefix) vendor/sqlite3ext.h sqlite-predict.h
	emcc -std=c99 -O1 -Ivendor/ -I./ -DSQLITE_CORE -DSQLITE_PREDICT_STATIC \
	  -DSQLITE_STRICT_SUBTYPE=1 \
	  -sALLOW_MEMORY_GROWTH -sEXIT_RUNTIME=1 -sSTACK_SIZE=1048576 \
	  tests/soak.c $(OBJS) vendor/sqlite3.c -o $(prefix)/soak.js

clean:
	rm -rf $(prefix) sqlite-predict.h

format:
	clang-format -i sqlite-predict.c predict-*.c

.PHONY: loadable loadable-onnx loadable-onnx-gpu debug test test-loadable \
  test-onnx test-asan-onnx clean format
