#!/bin/bash

export MY_LIB_PATH=/home/work/daiwenkai/Latest/tools/

export JAVA_HOME=${MY_LIB_PATH}/jdk1.8.0_152
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH

BAZEL_HOME=${MY_LIB_PATH}/
export PATH=${BAZEL_HOME}:$PATH
export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH


function build_practice()
{
    cd practice
    
    bazel clean --expunge
    sh -x run.sh
}

function main()
{
    build_practice
    [[ $? -ne 0 ]] && exit 1
    return 0
}

main 2>&1 
