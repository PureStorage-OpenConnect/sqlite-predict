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

OBJS=sqlite-predict.c predict-forecast.c predict-receipts.c vendor/sha256.c

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

clean:
	rm -rf $(prefix) sqlite-predict.h

format:
	clang-format -i sqlite-predict.c predict-*.c

.PHONY: loadable debug test test-loadable clean format
