#!/usr/bin/env bash
nosetests --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package="keras_global_self_attention" tests
