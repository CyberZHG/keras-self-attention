#!/usr/bin/env bash
pycodestyle --ignore=W605 --max-line-length=120 keras_self_attention tests
