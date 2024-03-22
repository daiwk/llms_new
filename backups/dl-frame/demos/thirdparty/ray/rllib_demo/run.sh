jumbo install glib
export LD_LIBRARY_PATH=~/.jumbo/lib/:$LD_LIBRARY_PATH 

#first, change /home/work/daiwenkai/python-3.6.9-gcc82/lib/python3.6/site-packages/ray/services.py
#### add
###cmd_prefix = "/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib".split(" ")
###then 
###
###    process = subprocess.Popen(
###        cmd_prefix + command,
###        env=modified_env,
###        cwd=cwd,
###        stdout=stdout_file,
###        stderr=stderr_file)


#~/daiwenkai/python-3.6.9-gcc82/bin/python3 rllib_demo.py 
#~/daiwenkai/python-3.6.9-gcc82/bin/python3 custom_env.py 
~/daiwenkai/python-3.6.9-gcc82/bin/python3 custom_env_1.py 
