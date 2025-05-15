#!/bin/bash

# If not root
if [ "$(whoami)" != "root" ]; then

    # Set RepositoryName if not already set
    if [[ -z "$RepositoryName" ]]; then
        export RepositoryName=$(basename $(git rev-parse --show-toplevel 2>/dev/null) || ls -1t /workspaces | tail -1)
        export LOCAL_WORKSPACE_FOLDER="/workspaces/$RepositoryName"
    fi

    # Minimal prompt with Git branch
    PS1='$(if [ -d .git ]; then echo -n "\[\033[32m\]($(git branch 2>/dev/null | grep '^*' | colrm 1 2))\[\033[00m\] "; fi)\w\$ '

    # Alias BFG
    alias bfg="java -jar /opt/share/bfg-1.14.0.jar"

    # Rewrite URLs in Flask and HTTP-Server outputs
    _hostname() {
        if [[ "$CODESPACES" == "true" ]]; then
            sed -E "s#http://[^:]+:(\x1b\[[0-9;]*m)?([0-9]+)(\x1b\[[0-9;]*m)?#https://${CODESPACE_NAME}-\2.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}#"
        else
            cat
        fi
    }

    flask() {
        command flask "$@" --host=127.0.0.1 2> >(_hostname >&2)
    }

    http-server() {
        command http-server "$@" | _hostname
    }
fi
