# SSH keys

## New key

```bash
ssh-keygen -t ed25519 -C "laptop-2024"
```

Use a passphrase. The agent remembers it, so you only type it once per session:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

On macOS, add `UseKeychain yes` to the config below so it survives restarts.

## Log into a server without a password

```bash
ssh-copy-id deploy@203.0.113.10
```

Once the key works, disable password login on the server (`PasswordAuthentication no` in sshd_config). Keep a second session open while testing so you cannot lock yourself out.

## Config file

`~/.ssh/config` saves a lot of typing:

```
Host homelab
    HostName 192.168.1.40
    User pi

Host github-work
    HostName github.com
    IdentityFile ~/.ssh/id_ed25519_work
```

With the second entry, clone work repositories with `git@github-work:org/repo.git` to use the work key.
