Dependencies:

Glad uses Python's Jinja template library to generate OpenGL headers. 

* Python: On Windows can be installed from Microsoft Store
* Jinja2: pip install jinja2

Once the repository is cloned:

```
cd dependencies
git submodule add https://github.com/vug/graphics-app-boilerplate.git
git submodule update --init --recursive
```

Start a new app:

* copy-paste one of the example apps
* rename `TARGET` in CMakeLists.txt
