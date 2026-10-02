# Git recipes

Things I look up every few months.

## Undo the last commit but keep the changes

```bash
git reset --soft HEAD~1
```

The changes stay staged. Use `--mixed` (the default) to keep them but unstaged. Only do this before pushing; for a pushed commit use `git revert <sha>` instead, which adds a new commit that undoes the old one.

## Fix the last commit message

```bash
git commit --amend -m "Better message"
```

## Get back a deleted branch

The reflog remembers where HEAD has been, even after a branch is gone:

```bash
git reflog
git switch -c recovered-branch <sha-from-reflog>
```

## Find which commit broke something

```bash
git bisect start
git bisect bad              # the current commit is broken
git bisect good v2.3.0      # this tag was fine
# test, then mark each step with git bisect good or git bisect bad
git bisect reset
```

With a test script that exits non-zero on failure, `git bisect run ./test.sh` does the whole search automatically.

## Park unfinished work

```bash
git stash push -m "half-done refactor"
git stash list
git stash pop
```

## Clean up a messy branch before review

```bash
git rebase -i main
```

Mark commits as `fixup` to fold them into the previous one. Never rebase a branch someone else is working on.
