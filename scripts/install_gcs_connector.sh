#!/usr/bin/env bash
set -e

user_site_packages=$(python -c 'import site; print(site.getusersitepackages())')
lib_dir="${user_site_packages}/pyspark/jars"
cd "${lib_dir}"
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar
