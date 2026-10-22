Before working on this branch, include your previous work from steps 1 and 2 (which have been merged to the main branch) in your branch:

```bash
git checkout 3/use_prefect
git pull origin master
git push
```

> Note: you can also use a `rebase` if you are familiar with this command.
The goal of this PR is to:

- [x] Use the prefect library to adapt the code developed in the PR2 to use `flow` and `task` objects
- [x] Create a deployment to retrain the model regularly

Files to be modified: the `src/modelling` folder

- [x] Add Prefect to the project dependencies and lockfile. This repository uses `pyproject.toml` and `uv.lock` instead of `requirements.in` and `requirements-dev.in`.
- [x] Update the README template with the steps to reproduce and run the code
- [x] Add instructions to run the Prefect UI and visualize flow runs

___

*TODO: delete this markdown file before merging the pull request*
