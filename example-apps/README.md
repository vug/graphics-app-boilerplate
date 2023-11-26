# Hi!

# 07 - Outlines via "Growth"

An alternative to the previous app. This time the outline is done via

* Draw the scene to screen
* Draw highlighted objects offscreen with a solid color shader
* Blit above texture into another offscreen one, via a shader that makes everywhere transparent except the pixels whose colored neighbor count N satisfies 0 < N < 8.
* Blit above texture to screen with blending on

With this technique no artifacts due to scaling up models occurs!

![image](screenshots/07-outlines-via-grow.png)

# 06 LearnOpenGL - Stencil Testing for Outlines

Implementation of [LearnOpenGL \- Stencil testing](https://learnopengl.com/Advanced-OpenGL/Stencil-testing) in graphics-app-boilerplate.
Observe the uneven thickness of the outline.

![image](screenshots/06-learnopengl-stencil-testing.png)

# 05 Sand Automata via Margolus Neighborhoods

Cellular automata for 2D falling sand simulation.

![image](screenshots/05-margolus-automata-sand.png)

See following for explanations

* [Block cellular automaton \- Wikipedia](https://en.wikipedia.org/wiki/Block_cellular_automaton)
* [Reversible cellular automata simulator: help page](https://dmishin.github.io/js-revca/help.html)

# 04 CUDA Mandelbrot

Draws Mandelbrot and Julia sets via CUDA. Can pan and zoom-in.

![image](screenshots/04-cuda-mandelbrot-1.png)

When using 32-bit floats the zoom-in limit is about 10^-6. (height side corresponds an interval of 4.1e-6)
![image](screenshots/04-cuda-mandelbrot-2.png)

Using doubles (64-bit floating points) fixes the discritization/pixellation problem at the same zoom-level
![image](screenshots/04-cuda-mandelbrot-3.png)

With doubles we can go down way further. Here height side corresponds to an interval of 3.1e-14 or something.
![image](screenshots/04-cuda-mandelbrot-4.png)

Enjoy zooming in and out multiple orders of magnitude
![image](screenshots/04-cuda-mandelbrot-5.gif)

# 03 First CUDA app

First CUDA app prepared via CMake. 
Project structure is such that CUDA code is compiled into a static library.
Which later is linked to C++ app code.

Introduction `launchKERNEL_NAME` code organization in `kernels.h` so that C++ does not involve any C++ CUDA extensions such as `<<<n,m>>>` syntax that causes an error to be reported in Visual Studio.

![image](screenshots/03-cuda-start.png)

# 02 Triangle (Without VBO)

Draws The Triangle without any meshes, just shaders

![image](screenshots/02-triangle-without-vbo.png)

# 01 Clear Window

Simplest app, just clear the window with a solid color which can be set via an ImGui ColorPicker

![image](screenshots/01-clear.png)









