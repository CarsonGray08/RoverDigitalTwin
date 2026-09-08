THEOS ?= $(HOME)/theos
ARCHS = arm64
TARGET = iphone:clang:latest:14.0
# Produce an installable .ipa (default Theos output is a .deb). Sideloader
# consumes this .ipa directly. See README.md in this directory.
PACKAGE_FORMAT = ipa

include $(THEOS)/makefiles/common.mk

APPLICATION_NAME = LiveScanner

LiveScanner_FILES = $(wildcard LiveScanner/*.m)
LiveScanner_FRAMEWORKS = UIKit ARKit SceneKit CoreVideo CoreImage CoreMotion
LiveScanner_CFLAGS = -fobjc-arc

include $(THEOS)/makefiles/application.mk
