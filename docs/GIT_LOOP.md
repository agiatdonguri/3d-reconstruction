# Local ↔ Trinity git loop

You develop on your Mac, push to GitHub, pull on Trinity. **You** commit/push; agents never commit.

## One-time: SSH key **on Trinity** for GitHub

On `trinity-2-28` (or any node — `/home` is shared):

```bash
# 1) make a key (press Enter for defaults; empty passphrase is fine for cluster convenience)
ssh-keygen -t ed25519 -C "tnishida-trinity" -f ~/.ssh/id_ed25519_github

# 2) show the public key — copy the whole line
cat ~/.ssh/id_ed25519_github.pub
```

On GitHub (browser):
1. github.com → your profile → **Settings** → **SSH and GPG keys** → **New SSH key**
2. Title: `trinity`
3. Paste the `.pub` line → Save

Still on Trinity, tell SSH to use that key for GitHub:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# test — should say "Hi agiatdonguri! You've successfully authenticated..."
ssh -T git@github.com
```

## One-time: clone into home

```bash
# if you already have a broken/empty folder, remove or rename it first
cd ~
rm -rf ~/3d-reconstruction   # only if it's empty/failed clone junk
git clone git@github.com:agiatdonguri/3d-reconstruction.git ~/3d-reconstruction
cd ~/3d-reconstruction
git status
```

## Daily loop

**On Mac** (after you edit / I edit files under `Documents/3d-reconstruction`):

```bash
cd ~/Documents/3d-reconstruction
git status
git add -A
git commit -m "your message"
git push origin main
```

**On Trinity** (any node):

```bash
cd ~/3d-reconstruction
git pull origin main
```

Then run jobs from that updated tree (or sync scene → `/scratch` and train).

## Notes

- Mac remote may use host alias `github-agiatdonguri`; Trinity can use plain `github.com` with the `~/.ssh/config` above.
- Keep heavy data out of git (`/data3`, `/scratch`) — only code in the repo.
- Train on a compute node from `/scratch`; code stays in `~/3d-reconstruction`.
