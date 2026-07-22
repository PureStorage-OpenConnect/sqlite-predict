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
endif

CC?=gcc
prefix?=dist

TARGET_LOADABLE=$(prefix)/predict0.$(LOADABLE_EXTENSION)

OBJS=sqlite-predict.c predict-forecast.c predict-receipts.c predict-tabular.c vendor/sha256.c

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

debug: $(prefix) vendor/sqlite3ext.h sqlite-predict.h $(OBJS)
	$(CC) -fPIC -shared -std=c99 -Wall -Wextra -Ivendor/ -I./ -g -O0 -DSQLITE_PREDICT_DEBUG $(CFLAGS) $(OBJS) -o $(TARGET_LOADABLE) $(LDFLAGS)

test-loadable: loadable
	cd tests && uv run pytest -q

test: test-loadable

# ASan+UBSan on the C soak driver (standalone executable: no DYLD
# injection needed, macOS SIP strips it for system binaries anyway).
# Covers every operation, receipts, replay, and the error paths.
test-asan: vendor/sqlite3ext.h sqlite-predict.h
	mkdir -p $(prefix)
	clang -std=c99 -g -O1 -fsanitize=address,undefined \
	  -fno-omit-frame-pointer -fno-sanitize-recover=undefined \
	  -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -Ivendor/ -I./ \
	  tests/soak.c $(OBJS) vendor/sqlite3.c -o $(prefix)/soak-asan
	UBSAN_OPTIONS=print_stacktrace=1 ./$(prefix)/soak-asan

# libFuzzer harness (statically links sqlite3.c; SQLITE_CORE build)
fuzz-build: vendor/sqlite3ext.h sqlite-predict.h
	mkdir -p $(prefix)
	clang -std=c99 -g -O1 -fsanitize=fuzzer,address,undefined \
	  -DSQLITE_CORE -DSQLITE_PREDICT_STATIC -Ivendor/ -I./ \
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

# real valgrind, in a Linux container (also exercises gcc + glibc)
test-valgrind:
	docker run --rm -v $$(pwd):/src -w /src gcc:13 bash -c "\
	  apt-get update -qq && apt-get install -y -qq valgrind unzip curl python3 > /dev/null && \
	  make clean && make loadable CC=gcc && \
	  gcc -std=c99 -g -O0 -Ivendor/ -I./ -DSQLITE_CORE -DSQLITE_PREDICT_STATIC \
	    tests/soak.c $(OBJS) vendor/sqlite3.c -o dist/soak -lm -lpthread -ldl && \
	  valgrind --leak-check=full --error-exitcode=9 --errors-for-leak-kinds=definite \
	    ./dist/soak"

clean:
	rm -rf $(prefix) sqlite-predict.h

format:
	clang-format -i sqlite-predict.c predict-*.c

.PHONY: loadable debug test test-loadable clean format
