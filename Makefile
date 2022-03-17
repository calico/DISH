HOST=127.0.0.1
TEST_PATH=./tests

PROJECT_NAME=MyProject
PROJECT_NEW_NAME=MyProject
FILES_WITH_PROJECT_NAME=Makefile .github/workflows/build-docs.yml README.md docs/index.md mkdocs.yml setup.py 
FILES_WITH_PROJECT_NAME_LC=Makefile README.md tests/test_myproject.py
PROJECT_NAME_LC=$(shell echo ${PROJECT_NAME} | tr '[:upper:]' '[:lower:]')
PROJECT_NEW_NAME_LC=$(shell echo ${PROJECT_NEW_NAME} | tr '[:upper:]' '[:lower:]')

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

rename-project:
	@echo renaming "${PROJECT_NAME}" to "${PROJECT_NEW_NAME}"
	@sed -i '' -e "s/${PROJECT_NAME}/${PROJECT_NEW_NAME}/g" ${FILES_WITH_PROJECT_NAME}
	@echo renaming "${PROJECT_NAME_LC}" to "${PROJECT_NEW_NAME_LC}"
	@sed -i '' -e "s/${PROJECT_NAME_LC}/${PROJECT_NEW_NAME_LC}/g" ${FILES_WITH_PROJECT_NAME_LC} && \
        git mv src/${PROJECT_NAME_LC} src/${PROJECT_NEW_NAME_LC} && \
	git mv tests/test_${PROJECT_NAME_LC}.py tests/test_${PROJECT_NEW_NAME_LC}.py
	@echo Project renamed
	
lint:
	flake8 --exclude=.tox

test: 
	pytest --verbose --color=yes $(TEST_PATH)

