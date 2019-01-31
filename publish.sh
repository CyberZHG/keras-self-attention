#!/usr/bin/env bash
rm -f dist/* && python3 setup.py sdist && twine upload dist/*
