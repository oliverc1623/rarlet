# RARLET: Robust Adversarial Reinforcement Learning with External Tactics

* [Poetry](https://python-poetry.org/)
    * For dependency management, packaging, and publishing
* [Ruff](https://github.com/astral-sh/ruff)
    * For linting/formatting (it's FAST)
* GitHub Actions
    * For CI/CD
* [Pytest](https://docs.pytest.org/en/8.2.x/)
    * For testing
* [pre-commit](https://pre-commit.com/)
    * For pre-commit hooks
* [PyInvoke](http://www.pyinvoke.org/)
    * For task running, because I hate `make`


## To get started
1. [Create a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
1. Clone the new repo
2. If poetry isn't installed, [you need to install it](https://python-poetry.org/docs/#installation).  
3. terminal `cd` into the project
3. run `poetry install`
3. Run `poetry run inv setup`
    

The setup will  
* Setup the poetry environment (or use the existing one you're activated to)
* Install the dependencies
* Setup the pre-commit hooks
* Ask you for some project details (name, author, etc) and update the pyproject.toml file

## Features
This sets up a basic set of checks to run.  If you already have a virtual environment setup for this project, you can skip all the `poetry run` parts of the command as long as that environment is active. Example `poetry run inv checks` would be `inv checks` if the environment is active. I won't be putting `poetry run` in front of every command, but if you don't have a virtual environment setup, you'll need to run `poetry run inv checks` instead of `inv checks`.

### Pre-commit
Pre-commit is used to run checks before you commit.  You can run `pre-commit run --all-files` to run all the checks.  The configuration for pre-commit is located in [.pre-commit-config.yaml](.pre-commit-config.yaml). If you find yourself unable to commit, this is almost certainly the reason. You need to install it for it to work on the client side. You can do this by running `pre-commit install`.

### Ruff
Ruff is used for linting and formatting. You can run 
`ruff check --fix` to check and fix the code. `ruff format` will only format the code.
The configuration for ruff is located in [ruff.toml](ruff.toml).

### Pytest
Pytest is used for testing.  You can run `pytest` to run all the tests. The CI is configured to run `pytest -m "not skipci"` so any test marked with `@pytest.mark.skipci` will not run in the CI pipeline. See [main_test.py](tests/main_test.py) for an example of how to use this.

### PyInvoke
PyInvoke is used for task running, and chosen because make is black magic to me.  You can run `inv --list` to see all the available tasks.  The tasks are located in the [tasks](tasks) folder.  The checks task will run all the checks. 

* `inv --list` will show you all the available tasks
* `inv checks` will run all the checks located in [checks.py](tasks/checks.py)

### GitHub
For pull requests, the pipeline will run `inv checks` and run all the formatting checks.  It will run the all the pytests, let you know what fails and succeeds in the pull request itself as well as give you a code coverage report.  The pipeline is located in [.github/workflows/ci.yml](.github/workflows/ci.yml).  

All pytests marked `@pytest.mark.skipci` will not run in the pipeline.  This is useful for tests that are slow, or require a specific environment to run.  You can run these tests locally, but they will not run in the pipeline.  You can see an example of this in [main_test.py](tests/main_test.py).

In order to get true coverage numbers in your report, the checks look for files in the src folder with a matching `_test.py` file in the `tests` folder.  If it doesn't have one, it creates a skeleton to just import.  
For example, [main.py](python_template/main.py) has a matching [main_test.py](tests/main_test.py) file.  

Theres also [issue templates](.github/ISSUE_TEMPLATE/bug_report.yml) and [rulesets](.github/rulesets/Require-Merge-Request.json) for the repository.  

## Contributing
If you have any suggestions, please open an issue.  If you'd like to contribute, please open a pull request.  I'm always looking for ways to improve this template. I'm open to suggestions, but I'm also very opinionated.  I'm trying to keep it as simple as possible while remaining good enough for production code.

## Updating from template
If you want to update your project from the template, or add the template to an existing project. 
There's a handy inv task. Just run `inv setup.update-from-template`.

or you can do it manually with the following commands

```bash
git remote add template https://github.com/lite-dsa/python-template.git
git fetch template
git merge template/main --allow-unrelated-histories
```


# Installation of VerifAI and Scenic

1. Clone the [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) repository and [Scenic](https://github.com/BerkeleyLearnVerify/Scenic) version 2.1.0.
2. Use python 3.8, higher versions of python might produce conflicts within some of the used libraries. 
3. Install both repositories, first Scenic then VerifAI. Go to their folders and run `python -m pip install -e` (we recommend installing everything in a virtual environment)
4. Download [Carla](https://carla.org/) (versions 0.9.12-0.9.15 work) 
5. Set the environmental variables of carla and its wheel python file.
6. Our experiments use `Town06` so make sure you install the additional maps for Carla.
7. Download this repository.

# Running the experiments

1. Activate the virtual environment where Scenic and VerifAI are installed.
2. Open Carla simulator 
3. To run an experiment run the python script `falsifier.py` with parameters: `--model carla` `--path path/to/scenario` eg: `--path scenarios-ddas/persistent_attack.scenic` `--e output_name` (to name the file where the falsification table will be stored) `--route folder` (to create a new folder to save the results of your simulation

# Additional notes

- Take into account the variable `inter_vehicle_distance` in the scenario (`.scenic` file) to specify the setpoint distance, this gives the distance between the vehicles (remember that the distances are measured from the center of mass so in reality the bumper to bumper distance is x - 4.95)
- Remember to change the variable `verifaiSampleType` in each scenario, our experiments have test `bo`(Bayesian Optimization)  and `ce` (Cross Entropy) as of Feb/2025
- `git submodule update --init` to intialize and update Scenic submodule in you're using the Scenic Gym branch
