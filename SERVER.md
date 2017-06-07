# SERVER MANAGEMENT
This is the management guide for server installation.

## Welcome message
Install figlet
```bash
$ sudo apt-get install figlet
```

```bash
$ sudo vi /etc/bash.bashrc

# Press [Shift] + [G] and write the code on the bottom
clear
printf "Welcome to Ubuntu 16.04.5 LTS (GNU/Linux-Mint-18 x86_64)\n"
printf "This is the server for the wide-residual-network implementation.\n\n"
printf " * Documentation: https://github.com/meliketoy/wide-residual-network\n\n"
printf "##############################################################\n"
figlet -f slant "Bumsoo Kim"
printf "\n\n"
printf " Data Mining & Information System Lab\n"
printf " GPU Computing machine : bumsoo@163.152.163.10\n\n"
printf " Administrator   : Bumsoo Kim\n"
printf " Please read the document\n"
printf "    https://github.com/meliketoy/wide-residual-network/README.md\n"
printf "##############################################################\n\n"
```

## Remote Server control

### 1. SCP call

```bash

# Upload your local file to server
$ scp -P 22 <LOCAL_DIR>/file [:username]@[:server_ip]:<SERVER_DIR>

# Download the server file to your local
$ scp -P 22 [:username]@[:server_ip]:<SERVER_DIR>/file <LOCAL_DIR>

# Upload your local directory to server
$ scp -P 22 -r <LOCAL_DIR>/file [:username]@[:server_ip]:<SERVER_DIR>

# Download the server file to your local
$ scp -P 22 -r [:username]@[:server_ip]:<SERVER_DIR>/file <LOCAL_DIR>

```

### 2. Save sessions by name
```bash
$ sudo vi ~/.bashrc

# Press [Shift] + [G] and enter the function on the bottom.

function server_func() {
    echo -n "[Enter the name of server]: "
    read server_name
                
    # echo "Logging in to server $server_name ..."
    if [ $server_name == [:servername] ]; then
        ssh [:usr]@[:ip].[:ip].[:ip].[:ip]
    fi
}
alias dmis_remote=server_func
```

## Github control

```bash
$ sudo vi ~/.netrc

machine github.com
login [:username]
password [:password]
```

### Jupyter notebook configuration

For jupyter notebook configuration, type in the command line below.
```bash
$ jupyter notebook --generate-config

* Result :
Writing default config to: <HOME_DIR>/.jupyter/jupyter_notebook_config.py

$ vi ~/.jupyter/jupyter_notebook_config.py
```

presh [Esc], then enter /ip to find the ip configuration. You will find the line below
``` bash
## The IP address the notebook server will listen on.
#c.NotebookApp.ip = 'localhost'
```

Erase the '#' and change it into ...
```bash
c.NotebookApp.ip = '163.152.163.112' # the ip address for your server
```

presh [Esc], then enter /port to find the port number. You will find the line below
```bash
## The port the notebook server will listen on.
#c.NotebookApp.port = 8888
```

Erase the '#' and enter whatever port number you want
```bash
c.NotebookApp.port = 9999

```

Now, Enjoy!
